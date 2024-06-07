### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 9afd85ee-b9de-4629-b9a8-3ce6ea0f10db
begin
	import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([
		Pkg.PackageSpec(name="ClimaSeaIce", rev="jlk9/mutable-slabseaicemodel"),
        Pkg.PackageSpec(name="Enzyme", version="0.12.8"),
        Pkg.PackageSpec(name="KernelAbstractions", version="0.9.20"),
        Pkg.PackageSpec(name="Oceananigans", rev="jlk9/stabilize-tuple-in-permute-boundary-conditions"),
		Pkg.PackageSpec(name="GLMakie")
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

	using GLMakie

	include("./ice_ocean_interaction.jl")

	Enzyme.API.runtimeActivity!(true)
	Enzyme.API.looseTypeAnalysis!(true)
	Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Grids.AbstractGrid}) = true
	Enzyme.EnzymeRules.inactive_type(::Type{<:Oceananigans.Clock}) = true
end

# ╔═╡ 92d7f11a-6a68-4d2e-9a5f-ee0798528c43
md"# Ocean Sea-Ice Inverse Problem"

# ╔═╡ 32388534-3ac7-497d-a933-086100ff0c20
md"As a (very!) simplified model, suppose we have a circlular floe of ice floating on top of a body of water. The water has a steady horizontal current represented by functions $u$ and $v$, and its initial temperature $T$ varies depending on depth and horizontal location. Both diffusion and advection will affect the water's temperature, which will in turn cause temperature changes and encourage growth or melt in the ice.

We'll implement this model using two Julia packages: `Oceananigans.jl` for ocean-flavored fluid dynamics, and `ClimaSeaIce.jl` for ice thermodynamics and dynamics. Both of these packages are designed for use with GPUs in mind, but for this tutorial we will only use CPUs since our problem will be fairly small. Both of these packages were written by scientists at the Climate Modeling Alliance (CliMA: https://clima.caltech.edu), and thus use very similar conventions.

To differentiate this model, we'll use `Enzyme`, a tool that performs automatic differentiation (AD) of code. `Enzyme` operates at the LLVM level (low-level virtual machine), which makes it usable for several programming languages. Here we'll use its Julia bindings, from the package `Enzyme.jl`.

The following block activates an environment using Oceananigans, ClimaSeaIce, and Enzyme - for this tutorial we'll use specific versions of each. You might also see references in the output to `KernelAbstractions.jl`, a library for writing code kernels that can be run on the CPU or GPU (both Oceananigans and ClimaSeaIce use this) as well as a file named `ice_ocean_interaction.jl`, which includes a helper function that implements energy transfer between water and ice."

# ╔═╡ 3adeeda5-7db4-4d38-a0c4-2adee41c14e8
md"### Step 1: Set up ocean and sea ice grid

Both Oceananigans and ClimaSeaIce feature a `struct` called `grid`, which discretizes the model domain and stores some important information:
- The hardware architecture our model is running on (CPU or GPU).
- The coordinate system of our grid (we're using rectilinear coordinates).
- The spatial dimensions of our grid.
- The resolution in each dimension.
- The topology of the domain boundaries, i.e. whether boundaries are bounded or periodic.

Our ocean grid will have three dimensions labeled $x$, $y$, and $z$. Our ice grid will only have $x$ and $y$ coordinates, since our ice occupies only a single spatial layer on top of the ocean."

# ╔═╡ 071a8881-0be3-4052-9c28-bc76057b6b5a
begin
	# Here we will set some important model parameters. First we specify the architecture:
	arch = Oceananigans.CPU()
	
	# Next we set the spatial resolution of the problem. We'll keep this low so it can be run locally:
	Nx = Ny = 16
	Nz = 16
	
	# Here we set the horizontal and vertical distances of the problem:
	x = y = (-π, π)
	z = (-1.0, 0)
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
md"Here we can see key information for both our ocean grid (labelled just `grid`) and our ice grid. Note the ocean grid is of size 8 $\times$ 8 $\times$ 16, while the ice grid is of size 8 $\times$ 8 $\times$ 1. We set the domain the be $[-\pi,\pi] \times [-\pi,\pi] \times [-0.5, 0.5]$, but any spatial coordinates can be used. All domain boundaries are either bounded or flat (in the case of the ice vertical dimension)."

# ╔═╡ 31ea7ca7-9e45-4722-9282-a636823f9a4e
@show grid

# ╔═╡ a30d3476-fe00-4e2c-a786-8673a64bc8cf
@show ice_grid

# ╔═╡ 1819e8b4-2902-4d43-95ea-e2748513ba6b
md"### Step 2: Set up ocean and sea ice model objects

With our grids set up, we can now focus on the other aspects of the model. Our ocean model needs a diffusivity to handle the temperature tracer diffusing through the water - we'll use a vertical scalar diffusivity for this. We label this choice `diffusion`.

We also need a velocity field for our water body. Here we'll use a time-invariant function, showing the water flows at a consistent speed and direction over time. We name our $x$-direction velocity `u` and our $y$-direction velocity `v`. We also isolate the ocean surface velocities with the `view` function so they can interact with the ice (`view` doesn't create a new instance of the array in its argument - it gives you an additional reference to that array that only tracks the slice passed to it).

For the ice, we need variables to represent heat and salt flux, as well as the boundary condition of the bottom with the water. We're simplifying this model by assuming the water has a constant salinity everywhere, but Oceananigans has the ability to track salinity as a tracer like temperature.

(One non-physical limitation of this model is a lack of ice advection - in other words, the ocean velocity will not directly make the ice above it move. Ice dynamics are still a work in progress in ClimaSeaIce. However, the ocean velocity does advect the ocean temperature, which in turn affects the ice through thermodynamics.)

Note that all of these variables use our `grid` or `ice_grid` objects, to determine the resolution they need and architecture on which they are represented."

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
md"#### With our required variables initialized, we can now create our ocean and ice `models`!

Our ocean `model` uses the hydrostatic approximation - hence we construct a `HydrostaticFreeSurfaceModel`. In addition to supplying the diffusivity and velocity fields above, we also add a tracer `T` to represent our temperature, and the WENO advection scheme to describe how temperature is moved by the ocean velocity field. Since we're assuming a constant salinity in our water, we won't track salinity as an additional tracer.

Our ice `model` is a simple slab model with only one layer (it's often referred to as a zero-layer model, since most other ice models represent internal temperature or energy). In addition to the fluxes above, we also supply a starting salinity, internal heat flux, and top heat flux and boundary condition."

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

Now we have functioning ocean and sea ice `models`. Suppose we have the final temperature distribution and ice thickness, and wish to use these to reconstruct the initial ocean temperature. In a real project, we might have measured temperature and thickness data to use for our inverse problem. For this tutorial, we'll instead generate synthetic data values using a function called `ice_ocean_data`. For this function we'll run our ocean and ice `models` forward a number of time steps (called $n_{\text{max}}$) to produce our data values. At each step we call the `time_step!` function for both `ocean_model` and `ice_model`, along with an additional function called `ice_ocean_latent_heat!` that handles the energy exchange between the ice and water, plus resulting temperature changes in the water.

Note: `time_step!` is a general-purpose function implemented in both Oceananigans and ClimaSeaIce that automatically runs every model component that changes in a time step. In our case it handles tracer (temperature) advection and diffusion when  `ocean_model` is the argument, and changes in ice thickness when `ice_model` is the argument (energy fluxes from `ice_ocean_latent_heat!` are also factored in for time stepping with ice). It will automatically handle other model components as additional complexity is added (such as equations of state, coriolis force, or velocity advection in an ocean `model`). If you want to run ice and ocean `models` at a higher level you can also create a `simulation` struct in either Oceananigans or ClimaSeaIce and call the `run!` function, which will simply call `time_step!` for a specified number of iterations. However, `simulations` are not yet compatible with Enzyme, so for now we have to call `time_step!` directly.


The function `ice_ocean_data` produces both the initial temperature distribution of the water (what we aim to invert for) as well as the final water temperature distribution and initial and final ice thickness (what we are given for the inverse problem). We also have helper functions to set the initial water diffusivity, water temperature, and ice thickness."

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
	function set_initial_temps!(ocean_model)
	    amplitude   = Ref(1)
	    width       = 0.1
	    #Tᵢ(x, y, z) = amplitude[] * exp(-z^2 / (2width^2)  - (x^2 + y^2) / 0.05)
		Tᵢ(x, y, z) = amplitude[] * exp(-z^2 / (2width^2)) * sin((x^2 + y^2) / 0.05)
	
	    set!(ocean_model, T=Tᵢ)
	
	    return nothing
	end
	
	# Generates the "real" data from a stable diffusion run:
	function ice_ocean_data(ocean_model, ice_model, diffusivity, n_max)
	    
	    set_diffusivity!(ocean_model, diffusivity)
	    set_initial_temps!(ocean_model)
	    
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
	n_max  = 100
end

# ╔═╡ 0b04432b-0e5d-4d99-9808-0a1ad7765f72
md"Then we can call `ice_ocean_data` and get the true initial water temperature and ice thickness $T_0$ and $h_0$, as well as the true final temperature and thickness $T_n$ and $h_n$. We can use the `@show` macro on each of these fields to see their shapes (which should align with their corresponding grids) and some statistics, and also plot them."

# ╔═╡ 045b79be-be33-4f17-b142-5f5b82c9e1f1
T₀, Tₙ, h₀, hₙ = ice_ocean_data(ocean_model, ice_model, κ, n_max)

# ╔═╡ 023737b6-401f-4ebf-8295-a2d783400a40
md"Here we plot the initial temperature along the surface and a cross section through the depth. Note how it fluctuates in circles near the surface and becomes constant at greater depths."

# ╔═╡ 042ee0ec-58b2-470d-84f5-302c87a88281
begin
	@show T₀
	xpoints = xnodes(grid, Center())
	ypoints = ynodes(grid, Center())
	zpoints = znodes(grid, Center())

	figsize = (1000, 500)
	temp_color_range = (-3, 3)

	fig = Figure(size=figsize)
	
	axTtop = Axis(fig[1, 1], xlabel="x (m)", ylabel="y (m)", title="Initial water surface Temperature")
	axTside = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)", title="Initial water depth temperature (cross-section)")
	Colorbar(fig[1, 3], limits = temp_color_range, colormap = :bluesreds,
    label = "Temperature (⚪ C)")
	
	heatmap!(axTtop, xpoints, ypoints, T₀.data[1:Nx,1:Ny,Nz], colormap=:bluesreds, colorrange = temp_color_range)
	heatmap!(axTside, xpoints, zpoints, T₀.data[1:Nx,Int(Ny/2),1:Nz], colormap=:bluesreds, colorrange = temp_color_range)

	fig
end

# ╔═╡ aa51161a-f1b6-4389-a9f8-b5c3ac1bc63e
md"These are the same plots for the final temperature. We can see how the current distorted the temperature field via advection, and how the colder water at the middle of the surface started to diffuse."

# ╔═╡ d6e9261a-5455-4bdf-86cd-dc52d70411fa
begin
	@show Tₙ

	fign = Figure(size=figsize)
	
	axTntop = Axis(fign[1, 1], xlabel="x (m)", ylabel="y (m)", title="Final water surface Temperature")
	axTnside = Axis(fign[1, 2], xlabel="x (m)", ylabel="z (m)", title="Final water depth temperature (cross-section)")
	Colorbar(fign[1, 3], limits = temp_color_range, colormap = :bluesreds,
    label = "Temperature (⚪ C)")
	
	heatmap!(axTntop, xpoints, ypoints, Tₙ.data[1:Nx,1:Ny,Nz], colormap=:bluesreds, colorrange = temp_color_range)
	heatmap!(axTnside, xpoints, zpoints, Tₙ.data[1:Nx,Int(Ny/2),1:Nz], colormap=:bluesreds, colorrange = temp_color_range)

	fign
end

# ╔═╡ 39555d5c-b20e-4058-a0d0-846723560fbb
md"Lastly, we have plots of the initial and final ice thickness. There's a little melt along the egdes of the ice flow:"

# ╔═╡ 843067f7-034f-4483-bcec-0b99af8b0862
begin
	@show h₀, hₙ

	figh = Figure(size=figsize)
	ice_color_range = (0, 0.05)
	
	axH0 = Axis(figh[1, 1], xlabel="x (m)", ylabel="y (m)", title="Initial ice thickness")
	axHn = Axis(figh[1, 2], xlabel="x (m)", ylabel="y (m)", title="Final ice thickness")
	Colorbar(figh[1, 3], limits = ice_color_range, colormap = :grays,
    label = "Ice thickness (m)")
	
	
	heatmap!(axH0, xpoints, ypoints, h₀.data[1:Nx,1:Ny], colormap=:grays, colorrange = ice_color_range)
	heatmap!(axHn, xpoints, ypoints, hₙ.data[1:Nx,1:Ny], colormap=:grays, colorrange = ice_color_range)

	figh
end

# ╔═╡ 76727025-342d-4cec-b632-e66e2b655ff1
md"### Step 4: Run the forward model

With our `model` structs and intial data, we can now run the actual ice-ocean model for an inverse problem. The names here are a little confusing, but this forward model will use our ocean and ice `model` objects - all of our required data is stored in these two objects. Similar to in `ice_ocean_data`, here we run the forward model a total of $n_\text{max}$ time steps, calling the `time_step!` for both `ocean_model` and `ice_model`, along with `ice_ocean_latent_heat!`.

Given some initial guess for our temperature distribution, we will run the model forward the set number of time steps and compare our computed final ocean temperature and ice thickness with the synthetic temperature and thickness from `ice_ocean_data`. Note that we provide diffusivity and initial ice thickness $h_i$ as function arguments, too. If we want to invert for these parameters in addition to or instead of the initial temperature, we can.

For the comparison, we're creating a function $J$ that's the combined mean squared error of our final ocean temperatures and ice thicknesses. Smaller $J$ implies that we have a better initial guess of $T_i$, since it produces similar data to our synthetic results $T_n$ and $h_n$."

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
		##
	    for i = 1:Nx, j = 1:Ny, k = 1:Nz
	        J += (T[i, j, k] - Tₙ[i, j, k])^2
	    end
		##
		##
	    for i = 1:Nx, j = 1:Ny
	        J += (h[i, j] - hₙ[i, j])^2
	    end
		##
	    return J::Float64
	end
end

# ╔═╡ b5d3f15a-aafe-43c0-8c09-ca8e6c6cbc13
md"### Step 5: Set up the inverse problem parameters

So now we have data for the ocean temperature and ice thickness after $n_\text{max}$ time steps, respectively called $T_n$ and $h_n$. We have a forward model defined by the function `ice_ocean_model`, utilizing the `model` structs from Oceananigans and ClimaSeaIce. We have a parameter we wish to invert for - the initial ocean temperature $T_i$ (we also have the 'true' ocean temperature $T_0$ used to generate our synthetic data, but in a real inverse problem we wouldn't already have that). And lastly, our forward model produces an output $J$ that measures its accuracy relative to $T_n$ and $h_n$. $J$ might be referred to as a misfit or loss function.

The next steps are to set up and run the inverse problem. We start with an initial guess for the starting temperature $T_i$. We call `ice_ocean_model` with this guess as an argument to produce $J$. A classic way to improve our guess for $T_i$ is by a method called gradient descent: if we somehow have the gradient for $J$ in terms of $T_i$, then we can adjust our guess for $T_i$ in the negative direction of the gradient to get an improved guess,

$$[T_i]_{n+1} = [T_i]_n - \gamma \nabla J\left([T_i]_n\right),$$

which will often produce a better approximation of $T_i$. The learning rate $\gamma$ determines the size of the step we take at each iteration of gradient descent - smaller steps require more iterations, but bigger steps run the risk of 'overshooting` and producing a new guess that performs worse than expected. Fortunately, we can choose an optimal $\gamma$ for our model and inverse problem.

The main challenge with this optimization is actually acquiring the gradient $\nabla J\left([T_i]_n\right)$. But this is made trivial with the use of Enzyme! If we supply enzyme with a zero-valued array that matches the size and typing of $T_i$, we can automatically generate its numerical derivative.

First, let's set an initial guess for $T_i$ - a constant array of value -1 seems reasonable, assuming we know the average value of the actual $T_0$ is somewhat close to that. We'll also set $\gamma = 0.2$ and our total number of gradient descent steps to 80:"

# ╔═╡ e0b647ee-827f-4bce-9b0c-244b7e55af75
begin
	
	Tᵢ = -1.0 .+ zeros(size(ocean_model.tracers.T))
	
	set!(ice_model, h=h₀)
	
	learning_rate = 0.2
	max_steps = 80
end

# ╔═╡ 44400683-d766-4fe1-b53b-03f5d2f60776
heatmap(xpoints, ypoints, Tᵢ[1:Nx,1:Ny,Nz], colormap=:bluesreds, colorrange = (-5, 5))

# ╔═╡ b7abbc43-4af1-4d20-9d01-14b28dc63cc0
md"### Step 6: Run the inverse problem

We'll run the inverse problem in a for-loop with `max_steps` iterations. In each iteration we have to create shadows of our forward model arguments, which store their derivative information. It's important that these shadows are set to all zero values before trying to use AD - this can be a little tricky for complicated structs like `ocean_model` and `ice_model`, but if we make sure to set fields mentioned earlier like boundary conditions to 0 then we'll be good. Since the diffusivity $\kappa$ and time steps $n_\text{max}$ are both scalars, they don't require shadows to be generated.

Once all the required shadow variables are set, we can call Enzyme's `autodiff` function in reverse mode by setting the first argument to `Enzyme.reverse`. This will use reverse-mode AD to construct the gradients of $J$ with respect to each of the inputs of ice_ocean_model, storing the gradient data in their shadow variables which are passed along with the actual inputs through `Duplicated` pairings.

Since we're inverting for $T_i$, we're only interested in the shadow $dT_i$. We apply it in our gradient descent to obtain a better guess for $T_i$, then repeat for `max_steps` iterations. At each step we'll look at the relative error between it and the true initial temperature $T_0$ to see how good our guess actually is.

*NOTE*: it might take a while for the following code block to run. Since Julia uses just-in-time compilation, complicated functions such as Enzyme's `autodiff` can be slow to compile. But after you finish differentiating the first iteration of gradient descent, subsequent iterations should be output very quickly.
"

# ╔═╡ 35cdbc37-bff5-4f6c-a691-3feb984fe148

# Update our guess of the initial tracer distribution, cᵢ:
for i = 1:max_steps
	# Create shadows and set all their values to 0:
    docean_model = Enzyme.make_zero(ocean_model)
    dice_model = Enzyme.make_zero(ice_model)
    dTᵢ = Enzyme.make_zero(Tᵢ)
    dTₙ = Enzyme.make_zero(Tₙ)
    dhᵢ = Enzyme.make_zero(h₀)
    dhₙ = Enzyme.make_zero(hₙ)
    set_diffusivity!(docean_model, 0)

    dice_model.heat_boundary_conditions = (top = PrescribedTemperature{Int64}(0), bottom = IceWaterThermalEquilibrium{ConstantField{Int64, 3}}(ConstantField(0)))
    dice_model.top_surface_temperature  = ConstantField(0)
    
    # Since we are only interested in duplicated variable Tᵢ for this run,
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
	
    
    @show i
    @show norm(dTᵢ)
    global Tᵢ .= Tᵢ .- (dTᵢ .* learning_rate)
    @show (norm(Tᵢ - T₀) / norm(T₀))
    
end


# ╔═╡ 399fde4f-d981-45e6-9258-04befb36d9fc
heatmap(xpoints, ypoints, Tᵢ[1:Nx,1:Ny,Nz], colormap=:bluesreds, colorrange = temp_color_range)

# ╔═╡ 934fb784-be1b-4b9d-ae5e-57ad07a38cfd
heatmap(xpoints, zpoints, Tᵢ[1:Nx,Int(Ny/2),1:Nz], colormap=:bluesreds, colorrange = temp_color_range)

# ╔═╡ 222416ae-1aa0-4f48-9e8d-3aabc34e21dd
heatmap(xpoints, ypoints, ocean_model.tracers.T.data[1:Nx,1:Ny,Nz], colormap=:bluesreds, colorrange = (-5, 5))

# ╔═╡ ca0ca8a0-1692-4cc0-8726-26de146051b7
md"### Results

Our first guess of constant -1 was *very* far off from the true initial temperature distribution, with a relative error of over 600%. After optimizing with gradient descent, our new guess has a much lower relative error of about 36%, but this is still a significant difference from the real value. Why?

First, inferring parameters for something as complicated as an ocean/ice model is difficult. Usually a model such as this is ill-conditioned in some way"

# ╔═╡ Cell order:
# ╟─92d7f11a-6a68-4d2e-9a5f-ee0798528c43
# ╟─32388534-3ac7-497d-a933-086100ff0c20
# ╠═9afd85ee-b9de-4629-b9a8-3ce6ea0f10db
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
# ╟─62b3adee-63b0-4f05-b304-d1d8b0d40ef9
# ╠═7cc53014-2395-4357-bbbb-d2d7eff59e20
# ╠═0b04432b-0e5d-4d99-9808-0a1ad7765f72
# ╠═045b79be-be33-4f17-b142-5f5b82c9e1f1
# ╟─023737b6-401f-4ebf-8295-a2d783400a40
# ╟─042ee0ec-58b2-470d-84f5-302c87a88281
# ╟─aa51161a-f1b6-4389-a9f8-b5c3ac1bc63e
# ╟─d6e9261a-5455-4bdf-86cd-dc52d70411fa
# ╟─39555d5c-b20e-4058-a0d0-846723560fbb
# ╟─843067f7-034f-4483-bcec-0b99af8b0862
# ╟─76727025-342d-4cec-b632-e66e2b655ff1
# ╠═66bef546-fe87-4d73-b6b5-9a305bff3765
# ╟─b5d3f15a-aafe-43c0-8c09-ca8e6c6cbc13
# ╠═e0b647ee-827f-4bce-9b0c-244b7e55af75
# ╠═44400683-d766-4fe1-b53b-03f5d2f60776
# ╟─b7abbc43-4af1-4d20-9d01-14b28dc63cc0
# ╠═35cdbc37-bff5-4f6c-a691-3feb984fe148
# ╠═399fde4f-d981-45e6-9258-04befb36d9fc
# ╠═934fb784-be1b-4b9d-ae5e-57ad07a38cfd
# ╠═222416ae-1aa0-4f48-9e8d-3aabc34e21dd
# ╠═ca0ca8a0-1692-4cc0-8726-26de146051b7
