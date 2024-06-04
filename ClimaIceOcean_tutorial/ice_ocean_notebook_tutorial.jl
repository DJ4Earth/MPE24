### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ ce9e8a0e-221e-11ef-187b-b505b61a430a
begin
	import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([
		Pkg.PackageSpec(name="ClimaSeaIce", rev="jlk9/mutable-slabseaicemodel"),
        Pkg.PackageSpec(name="Enzyme", version="0.12.8"),
        Pkg.PackageSpec(name="KernelAbstractions", version="0.9.19"),
        Pkg.PackageSpec(name="Oceananigans", rev="jlk9/stabilize-tuple-in-permute-boundary-conditions"),
    ])
	
	using Oceananigans
	using Oceananigans.TurbulenceClosures: with_tracers
	using Oceananigans.BoundaryConditions: fill_halo_regions!
	using Oceananigans.Fields: ZeroField, ConstantField
	using Oceananigans.Models.HydrostaticFreeSurfaceModels: tracernames
	
	using Enzyme
	using LinearAlgebra
	
	using ClimaSeaIce
	using ClimaSeaIce: melting_temperature
	using ClimaSeaIce.HeatBoundaryConditions: RadiativeEmission, IceWaterThermalEquilibrium

	using KernelAbstractions

	include("./ice_ocean_model.jl")

	Enzyme.API.runtimeActivity!(true)
	Enzyme.API.looseTypeAnalysis!(true)
	Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
	Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = true
end

# ╔═╡ 92d7f11a-6a68-4d2e-9a5f-ee0798528c43
md"# Ocean Sea-Ice Inverse Problem"

# ╔═╡ 32388534-3ac7-497d-a933-086100ff0c20
md"As a (very!) simplified model, suppose we have a block of ice floating on top of a body of water. The water is calm, with no current, but it has a temperature gradient which will diffuse over time. This will change the surface temperature of the water, which will in turn affect the growth/melt rate of the ice block.

We'll implement this model using two Julia packages: `Oceananigans.jl` for ocean-flavored fluid dynamics, and `ClimaSeaIce.jl` for ice thermodynamics and dynamics with salt. Both of these packages are designed for use with GPUs in mind, but for this tutorial we will only use CPUs since our problem will be fairly small. Both of these packages were written by scientists at the Climate Modeling Alliance (CliMA: https://clima.caltech.edu), and thus use very similar conventions.

To differentiate this model, we'll use `Enzyme`, a tool that performs automatic differentiation (AD) of code. `Enzyme` operates at the LLVM level (low-level virtual machine), which makes it usable for several programming languages. Here we'll use its Julia bindings, `Enzyme.jl`."

# ╔═╡ 3adeeda5-7db4-4d38-a0c4-2adee41c14e8
md"### Step 1: Set up ocean and sea ice grid

Both Oceananigans and ClimaSeaIce feature a `struct` called `grid`, which discretizes the model domain and stores some important information:
- The hardware architecture our model is running on (CPU or GPU).
- The coordinate system of our grid (we're using rectilinear coordinates).
- The spatial dimensions of our grid.
- The resolution in each dimension.
- The topology of the domain boundaries, i.e. whether boundaries are bounded or periodic.

Our ocean grid will have three dimensions labeled x, y, and z. Our ice grid will only have x and y coordinates, since our ice occupies only a single spatial layer on top of the ocean."

# ╔═╡ 071a8881-0be3-4052-9c28-bc76057b6b5a
begin
	# Here we will set some important model parameters. First we specify the architecture:
	arch = Oceananigans.CPU()
	
	# Next we set the spatial resolution of the problem. We'll keep this low so it can be run locally:
	Nx = Ny = 8 #64
	Nz = 16 #8
	
	# Here we set the horizontal and vertical distances of the problem:
	x = y = (-π, π)
	z = (-0.5, 0.5)
	topology = (Bounded, Bounded, Bounded)
	
	# Here we specify the grid - the spatial discretization of our model:
	grid = RectilinearGrid(arch, 
	                       size=(Nx, Ny, Nz);
	                       x, y, z, topology)
	
	# And here we set up the ice grid. Note that is is flat in the z-direction since all ice is at the surface:
	ice_grid = RectilinearGrid(arch,
	                           size = (Nx, Ny),
	                           topology = (topology[1], topology[2], Flat),
	                           x = x,
	                           y = y)
end

# ╔═╡ 9637e4af-6176-490d-9c96-c2d3b2a7b32d
md"Here we can see key information for both our ocean grid (labelled just `grid`) and our ice grid. Note the ocean grid is of size 8 $\times$ 8 $\times$ 16, while the ice grid is of size 8 $\times$ 8 $\times$ 1:"

# ╔═╡ 31ea7ca7-9e45-4722-9282-a636823f9a4e
@show grid

# ╔═╡ a30d3476-fe00-4e2c-a786-8673a64bc8cf
@show ice_grid

# ╔═╡ 1819e8b4-2902-4d43-95ea-e2748513ba6b
md"### Step 2: Set up ocean and sea ice model objects

With our grids set up, we can now focus on the other aspects of the model. Our ocean model needs a diffusivity to handle tracers like temperature diffusing through the water - we'll use a vertical scalar diffusivity for this. We lable this choice `diffusion`.

We also need a velocity field for our water body. Here we'll use a time-invariant function, showing the water flows at a consistent speed and direction over time. We name our $x$-direction velocity `u` and our $y$-direction velocity `v`. We also isolate the ocean surface velocities so they can interact with the ice.

For the ice, we need variables to represent heat and salt flux, as well as the boundary condition of the bottom with the water.

(One non-physical limitation of this model is a lack of ice advection - in other words, the ocean velocity will not directly make the ice above it move. Ice dynamics are still a work in progress in `ClimaSeaIce`. However, the ocean velocity does advect the ocean temperature, which in turn affects the ice through thermodynamics.)

Note that all of these variables use our `grid` or `ice_grid` objects, to determine the resolution and architecture on which they are represented."

# ╔═╡ 1853ac7c-b18a-43df-86a9-1f5195b19fda
begin
	# Then we set a maximal diffusivity and diffusion type:
	const maximum_diffusivity = 100
	diffusion = VerticalScalarDiffusivity(κ=0.1)
	
	# For this problem we'll have a constant-values velocity field - the water will flow at a
	# constant speed and direction over time. We have to assign the proper velocity values to
	# every point in our grid:
	u = XFaceField(grid)
	v = YFaceField(grid)
	
	U = 40
	u₀(x, y, z) = - U * cos(x + π/4) * sin(y)
	v₀(x, y, z) = + U * 0.5
	
	set!(u, u₀)
	set!(v, v₀)
	fill_halo_regions!(u)
	fill_halo_regions!(v)
	
	ice_ocean_heat_flux      = Field{Center, Center, Nothing}(ice_grid)
	top_ocean_heat_flux = Qᵀ = Field{Center, Center, Nothing}(ice_grid)
	top_salt_flux       = Qˢ = Field{Center, Center, Nothing}(ice_grid)
	
	bottom_bc = IceWaterThermalEquilibrium(ConstantField(30)) 
	
	ocean_surface_velocities = (u = view(u, :, :, Nz), #interior(u, :, :, Nz),
	                            v = view(v, :, :, Nz), #interior(v, :, :, Nz),    
	                            w = ZeroField())
end

# ╔═╡ dd2b04b7-d35d-42ad-8887-206da0576688
md"#### With our required variables initialized, we can now create our ocean and ice models!

Our ocean model uses the hydrostatic approximation - hence we construct a `HydrostaticFreeSurfaceModel`. In addition to supplying the diffusivity and velocity fields above, we also add a tracer `T` to represent our temperature, and the WENO advection scheme to describe how temperature is moved by the ocean velocity field. We'll assume a constant salinity in our water.

Our ice model is a simple slab model with only one layer.In addition to the fluxes above, we also supply a starting salinity, internal heat flux, and top heat flux and boundary condition."

# ╔═╡ a3efa97e-467b-43a2-87db-8c3bdc251d25
ocean_model = HydrostaticFreeSurfaceModel(; grid,
                            tracer_advection = WENO(),
                            tracers = :T,
                            buoyancy = nothing,
                            velocities = PrescribedVelocityFields(; u, v),
                            closure = diffusion)

# ╔═╡ 63f3f573-d12f-4c22-aeec-275c33750ab9
ice_model = SlabSeaIceModel(ice_grid;
                    velocities = ocean_surface_velocities,
                    advection = nothing,
                    ice_consolidation_thickness = 0.05,
                    ice_salinity = 4,
                    internal_heat_flux = ConductiveFlux(conductivity=2),
                    top_heat_flux = ConstantField(0), # W m⁻²
                    top_heat_boundary_condition = PrescribedTemperature(-10),
                    bottom_heat_boundary_condition = bottom_bc,
                    bottom_heat_flux = ice_ocean_heat_flux)

# ╔═╡ f8194371-97d4-45f4-8441-309bc535302c
md"### Step 3: Obtain the (fictional) real data values

Now we have a functioning ocean and sea ice model. Suppose we have the final temperature distribution and ice thickness, and wish to use these to reconstruct the initial ocean temperature. In a real project, we might have measured temperature and thickness data to use for our inverse problem. For this tutorial, we'll instead generate synthetic data values using a function called `ice_ocean_data`. For this function we'll run our ocean and ice models forward a number of time steps (called $n_{\text{max}}$) to produce our data values. The function `ice_ocean_data` produces both the initial temperature distribution of the water (what we aim to invert for) as well as the final ocean temperature distribution and ice thickness (what we are given for the inverse problem)."

# ╔═╡ fc39f605-9635-402d-b60a-1c8c15a82c89
begin
	# Set the initial ice thickness:
	function hᵢ(x, y)
	    if sqrt(x^2 + y^2) < π / 3
	        return 0.05 #0.05 + 0.01 * rand()
	    else 
	        return 0
	    end
	end
	
	set!(ice_model, h=hᵢ)
	
	# Sets the initial diffusivity:
	function set_diffusivity!(ocean_model, diffusivity)
	    closure = VerticalScalarDiffusivity(; κ=diffusivity)
	    names = tuple(:T) # tracernames(model.tracers)
	    closure = with_tracers(names, closure)
	    ocean_model.closure = closure
	    return nothing
	end
	
	# This produces an initial data distribution:
	function set_initial_data!(ocean_model)
	    amplitude   = Ref(1)
	    width       = 0.1
	    Tᵢ(x, y, z) = amplitude[] * exp(-z^2 / (2width^2)  - (x^2 + y^2) / 0.05)
	
	    set!(ocean_model, T=Tᵢ)
	
	    return nothing
	end
	
	# Generates the "real" data from a stable diffusion run:
	function ice_ocean_data(ocean_model, ice_model, diffusivity, n_max)
	    
	    set_diffusivity!(ocean_model, diffusivity)
	    set_initial_data!(ocean_model)
	    
	    # Do time-stepping
	    Nx, Ny, Nz = size(ocean_model.grid)
	    κ_max = maximum_diffusivity
	    Δz = 2π / Nz
	    Δt = 1e-1 * Δz^2 / κ_max
	    @show Δt
	
	    ocean_model.clock.time = 0
	    ocean_model.clock.iteration = 0
	
	    # Running one time step to "stabilize" model fields:
	    for n = 1:1
	        time_step!(ice_model, Δt)
	        ice_ocean_latent_heat!(ice_model, ocean_model, Δt)
	        time_step!(ocean_model, Δt; euler=true)
	    end
	
	    T₀ = deepcopy(ocean_model.tracers.T)
	    h₀ = deepcopy(ice_model.ice_thickness)
	
	    for n = 1:n_max
	        time_step!(ice_model, Δt)
	        ice_ocean_latent_heat!(ice_model, ocean_model, Δt)
	        time_step!(ocean_model, Δt; euler=true)
	
	        #@show n
	        #@show ocean_model.tracers.T
	        #@show ice_model.ice_thickness
	    end
	
	    # Compute scalar metric
	    Tₙ = deepcopy(ocean_model.tracers.T)
	    hₙ = deepcopy(ice_model.ice_thickness)
	
	    return T₀, Tₙ, h₀, hₙ
	end
end

# ╔═╡ 62b3adee-63b0-4f05-b304-d1d8b0d40ef9
md"Let's set diffusivity $\kappa = 1$ and our number of forward time steps $n_{\text{max}} = 10$:"

# ╔═╡ 7cc53014-2395-4357-bbbb-d2d7eff59e20
begin
	κ = 1
	n_max  = 10
end

# ╔═╡ 045b79be-be33-4f17-b142-5f5b82c9e1f1
T₀, Tₙ, h₀, hₙ = ice_ocean_data(ocean_model, ice_model, κ, n_max)

# ╔═╡ 042ee0ec-58b2-470d-84f5-302c87a88281
@show T₀

# ╔═╡ d6e9261a-5455-4bdf-86cd-dc52d70411fa
@show Tₙ

# ╔═╡ 843067f7-034f-4483-bcec-0b99af8b0862
@show h₀

# ╔═╡ d7d5d9ac-0f13-40fc-98be-0427f4863dc8
@show hₙ

# ╔═╡ 76727025-342d-4cec-b632-e66e2b655ff1
md"### Step 4: Run the forward model

With our model structs and intial data, we can now run the actual ice-ocean model for an inverse problem. Given some initial guess for our temperature distribution, we will run the model forward the set number of time steps and compare our computed final ocean temperature and ice thickness with the synthetic temperature and thickness from `ice_ocean_data`."

# ╔═╡ 66bef546-fe87-4d73-b6b5-9a305bff3765
begin
	function set_initial_condition!(ocean_model, Tᵢ)
	    set!(ocean_model, T=Tᵢ)
	    return nothing
	end
	
	# cᵢ is the proposed initial condition, cₙ is the actual data collected
	# at the final time step:
	function ice_ocean_model(ocean_model, ice_model, diffusivity, n_max, Tᵢ, Tₙ, hᵢ, hₙ)
	    set_diffusivity!(ocean_model, diffusivity)
	    set_initial_condition!(ocean_model, Tᵢ)
	
	    set!(ice_model, h=hᵢ)
	    
	    # Run the forward model:
	    Nx, Ny, Nz = size(ocean_model.grid)
	    κ_max = maximum_diffusivity
	    Δz = 2π / Nz
	    Δt = 1e-1 * Δz^2 / κ_max
	
	    ocean_model.clock.time = 0
	    ocean_model.clock.iteration = 0
	
	    for n = 1:n_max
	        time_step!(ice_model, Δt)
	        ice_ocean_latent_heat!(ice_model, ocean_model, Δt)
	        time_step!(ocean_model, Δt; euler=true)
	    end
	
	    T = ocean_model.tracers.T
	    h = ice_model.ice_thickness
	    # Compute the misfit of our forward model run with the true data Tₙ:
	    J = 0.0
	    for i = 1:Nx, j = 1:Ny, k = 1:Nz
	        J += (T[i, j, k] - Tₙ[i, j, k])^2
	    end
	    for i = 1:Nx, j = 1:Ny
	        J += (h[i, j] - hₙ[i, j])^2
	    end
	
	    return J::Float64
	end
end

# ╔═╡ b5d3f15a-aafe-43c0-8c09-ca8e6c6cbc13
md"### Step 5: Set up the inverse problem parameters"

# ╔═╡ e0b647ee-827f-4bce-9b0c-244b7e55af75
begin
	
	Tᵢ = -1.0 .+ zeros(size(ocean_model.tracers.T))
	
	set!(ice_model, h=h₀)
	
	learning_rate = 0.2
	max_steps = 80
	δ = 0.01
end

# ╔═╡ b7abbc43-4af1-4d20-9d01-14b28dc63cc0
md"### Step 6: Run the inverse problem"

# ╔═╡ 35cdbc37-bff5-4f6c-a691-3feb984fe148
# Update our guess of the initial tracer distribution, cᵢ:
for i = 1:max_steps
    docean_model = Enzyme.make_zero(ocean_model)
    dice_model = Enzyme.make_zero(ice_model)
    dTᵢ = Enzyme.make_zero(Tᵢ)
    dTₙ = Enzyme.make_zero(Tₙ)
    dhᵢ = Enzyme.make_zero(h₀)
    dhₙ = Enzyme.make_zero(hₙ)
    set_diffusivity!(docean_model, 0)

    dice_model.heat_boundary_conditions = (top = PrescribedTemperature{Int64}(0), bottom = IceWaterThermalEquilibrium{ConstantField{Int64, 3}}(ConstantField(0)))
    dice_model.top_surface_temperature  = ConstantField(0)

    @show dice_model.internal_heat_flux.func

    @show dice_model

    #set!(ice_model, h=h₀)
    
    # Since we are only interested in duplicated variable cᵢ for this run,
    # we do not use dJ here:
    
    dJ = autodiff(Enzyme.Reverse,
                    ice_ocean_model,
                    Duplicated(ocean_model, docean_model),
                    Duplicated(ice_model, dice_model),
                    Const(κ),
                    Const(n_max),
                    Duplicated(Tᵢ, dTᵢ),
                    Duplicated(Tₙ, dTₙ),
                    Duplicated(h₀, dhᵢ),
                    Duplicated(hₙ, dhₙ))
	
    @show ocean_model.tracers.T
    @show ice_model.ice_thickness
    
    @show i
    @show norm(dTᵢ)
    global Tᵢ .= Tᵢ .- (dTᵢ .* learning_rate)
    @show (norm(Tᵢ - T₀) / norm(T₀))

    # Stop gradient descent if dcᵢ is sufficiently small:
    #if norm(dTᵢ) < δ
    #    break
    #end
    
end

# ╔═╡ ca0ca8a0-1692-4cc0-8726-26de146051b7


# ╔═╡ Cell order:
# ╠═ce9e8a0e-221e-11ef-187b-b505b61a430a
# ╟─92d7f11a-6a68-4d2e-9a5f-ee0798528c43
# ╟─32388534-3ac7-497d-a933-086100ff0c20
# ╟─3adeeda5-7db4-4d38-a0c4-2adee41c14e8
# ╠═071a8881-0be3-4052-9c28-bc76057b6b5a
# ╟─9637e4af-6176-490d-9c96-c2d3b2a7b32d
# ╠═31ea7ca7-9e45-4722-9282-a636823f9a4e
# ╠═a30d3476-fe00-4e2c-a786-8673a64bc8cf
# ╟─1819e8b4-2902-4d43-95ea-e2748513ba6b
# ╠═1853ac7c-b18a-43df-86a9-1f5195b19fda
# ╟─dd2b04b7-d35d-42ad-8887-206da0576688
# ╠═a3efa97e-467b-43a2-87db-8c3bdc251d25
# ╠═63f3f573-d12f-4c22-aeec-275c33750ab9
# ╟─f8194371-97d4-45f4-8441-309bc535302c
# ╠═fc39f605-9635-402d-b60a-1c8c15a82c89
# ╠═62b3adee-63b0-4f05-b304-d1d8b0d40ef9
# ╠═7cc53014-2395-4357-bbbb-d2d7eff59e20
# ╠═045b79be-be33-4f17-b142-5f5b82c9e1f1
# ╠═042ee0ec-58b2-470d-84f5-302c87a88281
# ╠═d6e9261a-5455-4bdf-86cd-dc52d70411fa
# ╠═843067f7-034f-4483-bcec-0b99af8b0862
# ╠═d7d5d9ac-0f13-40fc-98be-0427f4863dc8
# ╟─76727025-342d-4cec-b632-e66e2b655ff1
# ╠═66bef546-fe87-4d73-b6b5-9a305bff3765
# ╠═b5d3f15a-aafe-43c0-8c09-ca8e6c6cbc13
# ╠═e0b647ee-827f-4bce-9b0c-244b7e55af75
# ╠═b7abbc43-4af1-4d20-9d01-14b28dc63cc0
# ╠═35cdbc37-bff5-4f6c-a691-3feb984fe148
# ╠═ca0ca8a0-1692-4cc0-8726-26de146051b7
