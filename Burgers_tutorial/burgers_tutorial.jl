### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 5748f296-c540-417f-b39d-f82d0e46559e
begin
	using Pkg
	Pkg.activate(mktempdir())
	Pkg.add(name="LLVM", version=v"7.1")
	Pkg.add("Adapt")
	Pkg.add("Enzyme")
	Pkg.add("KernelAbstractions")
	Pkg.add("Checkpointing")
	Pkg.add("Plots")
end

# ╔═╡ bbe566f7-02b7-480b-bd63-d6c72aa1ac40
begin
using KernelAbstractions
using Adapt
end

# ╔═╡ 06f23f52-0fcc-4707-b06c-80f4529b506d
using Enzyme

# ╔═╡ cd9ef5ba-5e59-43ed-82d1-588dc7effefe
using Checkpointing

# ╔═╡ a575949c-2368-11ef-2b2d-2fe3a2ffbae3
md"""
# 2D Burgers Equations in a Nutshell

## Governing equations

```math
\begin{align}
    \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} &= \nu \nabla^2 u\\ 
    \frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} &= \nu \nabla^2 v 
\end{align}
```
where $u$ and $v$ represent the $x$ and $y$ velocities of a fluid and $\nu$ is the viscosity coefficient.
"""

# ╔═╡ b96ea551-35ef-4b87-989b-fe5e49e098cc
const KA = KernelAbstractions

# ╔═╡ ddf67473-5f5f-44b2-bd40-415772f5aead
begin
mutable struct Burgers
    nextu::AbstractMatrix
    nextv::AbstractMatrix
    lastu::AbstractMatrix
    lastv::AbstractMatrix
    nx::Int # Grid points in x direction
    ny::Int # Grid points in y direction
    μ::Float64
    dx::Float64
    dy::Float64
    dt::Float64
    tsteps::Int
    backend::KA.Backend
end
function Burgers(
    Nx::Int,
    Ny::Int,
    μ::Float64,
    dx::Float64,
    dy::Float64,
    dt::Float64,
    tsteps::Int,
    backend=CPU(),
)
    return Burgers(
        adapt(backend,zeros(Nx, Ny)),
        adapt(backend,zeros(Nx, Ny)),
        adapt(backend,zeros(Nx, Ny)),
        adapt(backend,zeros(Nx, Ny)),
        Nx,
        Ny,
        μ,
        dx,
        dy,
        dt,
        tsteps,
        backend,
    )
end
end

# ╔═╡ feb5a4ce-f32e-4f72-94c9-a39460a6866e
md"""
To discretize the system, we use a centered finite difference scheme in space and an explicit forward Euler scheme in time.
"""

# ╔═╡ d7df76fd-dbb3-43bd-a1a9-35b69cebcbcb
@kernel function stencil_kernel!(lastu, nextu, lastv, nextv, dx, dy, dt, μ, nx, ny)
    i, j = @index(Global, NTuple)
    nextu[i+1, j+1] =
        lastu[i+1, j+1] +
        dt * (
            (
                -lastu[i+1, j+1] / (2 * dx) * (lastu[i+2, j+1] - lastu[i, j+1]) -
                lastv[i+1, j+1] / (2 * dy) * (lastu[i+1, j+2] - lastu[i+1, j])
            ) +
            μ * (
                (lastu[i+2, j+1] - 2 * lastu[i+1, j+1] + lastu[i, j+1]) / dx^2 +
                (lastu[i+1, j+2] - 2 * lastu[i+1, j+1] + lastu[i+1, j]) / dy^2
            )
        )
    nextv[i+1, j+1] =
        lastv[i+1, j+1] +
        dt * (
            (
                -lastu[i+1, j+1] / (2 * dx) * (lastv[i+2, j+1] - lastv[i, j+1]) -
                lastv[i+1, j+1] / (2 * dy) * (lastv[i+1, j+2] - lastv[i+1, j])
            ) +
            μ * (
                (lastv[i+2, j+1] - 2 * lastv[i+1, j+1] + lastv[i, j+1]) / dx^2 +
                (lastv[i+1, j+2] - 2 * lastv[i+1, j+1] + lastv[i+1, j]) / dy^2
            )
        )
end

# ╔═╡ dd62122a-89c0-471a-97c1-5f3d60b98f40
md"""
The equation is solved on a square domain, $(x, y) \in [-L, L] \times [-L, L]$, with the initial velocities
```math

u(0,x,y) = \exp\left(-x^2 - y^2 \right), \; \; \; \; v(0,x,y) = \exp\left(-x^2 - y^2\right).
```
Here we use the domain $[-3,3] \times [-3,3].$
"""


# ╔═╡ 0dc72edd-d11c-4f71-baee-a5c265b2d135
@kernel function set_ic_kernel!(lastu, nextu, lastv, nextv, nx, ny)
    i, j = @index(Global, NTuple)
    lastu[i+1, j+1] = exp(-(-3.0 + i * 6.0 / (nx - 1))^2 - (-3.0 + j * 6.0 / (ny - 1))^2)
    lastv[i+1, j+1] = exp(-(-3.0 + i * 6.0 / (nx - 1))^2 - (-3.0 + j * 6.0 / (ny - 1))^2)
    nextu[i+1, j+1] = exp(-(-3.0 + i * 6.0 / (nx - 1))^2 - (-3.0 + j * 6.0 / (ny - 1))^2)
    nextv[i+1, j+1] = exp(-(-3.0 + i * 6.0 / (nx - 1))^2 - (-3.0 + j * 6.0 / (ny - 1))^2)
end

# ╔═╡ 75f0cbda-57f4-4232-93f8-a88ccd764d16
md"""
We use Dirichlet conditions on all four boundaries
```math
u(t,x,-L) = u(t,x, L) = u(t,-L, y) = u(t,L, y) = 0.
```
"""

# ╔═╡ 02e8e500-9785-436c-8e2c-d3455ab8eec3
function set_bc!(burgers::Burgers)
    (nx, ny) = (burgers.nx, burgers.ny)
    burgers.lastu[1:1, 1:ny] .= 0.0
    burgers.lastv[1:1, 1:ny] .= 0.0
    burgers.lastu[nx:nx, 1:ny] .= 0.0
    burgers.lastv[nx:nx, 1:ny] .= 0.0
    burgers.lastu[1:nx, 1:1] .= 0.0
    burgers.lastv[1:nx, 1:1] .= 0.0
    burgers.lastu[1:nx, ny:ny] .= 0.0
    burgers.lastv[1:nx, ny:ny] .= 0.0
    return nothing
end

# ╔═╡ 69a5e411-7f1b-4f38-895d-c92a5fb152f7
md"""
At each time $t_f$ we can compute the current kinetic energy.
```math
    J = \frac{1}{N_x \cdot N_y} \sum_{j = 1}^{N_x} \sum_{k = 1}^{N_y} \left( u(t_f, x_j, y_k)^2+ v(t_f, x_j, y_k)^2 \right),
```
"""


# ╔═╡ a25e3d03-8eed-4a0b-9fd7-c5e30f9c8f1c
function energy(burgers::Burgers)
    @inbounds lenergy =
        sum(burgers.nextu[2:end-1, 2:end-1] .^ 2 .+ burgers.nextv[2:end-1, 2:end-1] .^ 2) /
        (burgers.nx*burgers.ny)
end

# ╔═╡ eabb6565-946f-4e00-a3ac-9ef33d05c594
md"""
Finally, get things moving by advancing in time 
"""

# ╔═╡ 664315e2-b4db-41ad-82d1-3039454fed1e
md"""
and returning the final kinetic energy after $T$ timesteps.
"""

# ╔═╡ 12dfa763-b307-4f76-9a0d-5b24e6130da9
md"""
How do we launch a kernel?

stencil_kernel!(backend)
"""

# ╔═╡ 64b6eca3-8b8a-470c-b496-8207d88fb99c
function stencil!(burgers::Burgers)
    stencil_kernel!(burgers.backend)(
        burgers.lastu,
        burgers.nextu,
        burgers.lastv,
        burgers.nextv,
        burgers.dx,
        burgers.dy,
        burgers.dt,
        burgers.μ,
        burgers.nx,
        burgers.ny,
        ndrange=(burgers.nx-2, burgers.ny-2),
    )
    KA.synchronize(burgers.backend)
end

# ╔═╡ 4b4f81f6-6ff0-430f-8a86-965d4a41044d
function advance!(burgers::Burgers)
    stencil!(burgers)
	copyto!(burgers.lastu, burgers.nextu)
    copyto!(burgers.lastv, burgers.nextv)
    return nothing
end

# ╔═╡ 9c21cfa8-0db3-4d74-9465-75bf25b15a5b
function final_energy!(burgers::Burgers)
    for i = 1:burgers.tsteps
        advance!(burgers)
    end
    return energy(burgers)
end

# ╔═╡ 96a54adb-015c-4243-a99d-9cda695f2a4d
function set_ic!(burgers::Burgers)
    set_ic_kernel!(burgers.backend)(
        burgers.lastu, burgers.nextu, burgers.lastv, burgers.nextv, burgers.nx, burgers.ny,
        ndrange=(burgers.nx-2, burgers.ny-2),
    )
    KA.synchronize(burgers.backend)
    return nothing
end

# ╔═╡ 48b5e47b-e090-401d-ad7a-4898874b5117
begin
Nx = 100
Ny = 100
tsteps = 10
μ = 0.01 # # U * L / Re,   nu
dx = 1e-1
dy = 1e-1
dt = 1e-3 # dt < 0.5 * dx^2
end

# ╔═╡ de82b316-f8b9-479d-acac-dd18768e1e43
burgers = Burgers(Nx, Ny, μ, dx, dy, dt, tsteps)

# ╔═╡ 32aadddf-6220-4e3d-aae1-09d58e173b41
set_ic!(burgers)

# ╔═╡ c5feb1d9-4703-49a2-bedc-b996887f4d91
set_bc!(burgers)

# ╔═╡ 50c2b95a-9e72-48b9-879b-d31a70c0f6cb
ienergy = energy(burgers)

# ╔═╡ bc294229-2b02-4b19-8f1b-71348793b323
function velocity_magnitude(burgers)
	burgers.nextu[2:end-1, 2:end-1] .^ 2 + burgers.nextv[2:end-1, 2:end-1] .^ 2
end

# ╔═╡ f9f6b93f-74b4-49bb-91ed-71b14c87fb3a
begin
using Plots
color = :balance
surface(
    range(-3, 3, length=burgers.nx-2),
    range(-3, 3, length=burgers.ny-2),
    velocity_magnitude(burgers);xlabel = "x", ylabel = "y", c = color,
    legend=:none
)
end

# ╔═╡ daf57c26-3d7c-48b1-a26f-904f1c39a112
fenergy = final_energy!(burgers)

# ╔═╡ f43f9eb8-902a-4489-981d-2c356d025289
surface(
    range(-3, 3, length=burgers.nx-2),
    range(-3, 3, length=burgers.ny-2),
    velocity_magnitude(burgers);xlabel = "x", ylabel = "y", c = color,
    legend=:none
)

# ╔═╡ b8201b3f-5187-4fdd-86a2-feb4e1d4f05b
md"""
Let's crank up the resolution
"""

# ╔═╡ a9814ad4-7c9b-4c25-8fa4-b1683ad9be82
begin
burgers_hd = Burgers(1000, 1000, μ, 1e-2, 1e-2, dt, tsteps)
set_ic!(burgers_hd)
set_bc!(burgers_hd)
final_energy!(burgers_hd)
surface(
	range(-3, 3, length=burgers_hd.nx-2),
	range(-3, 3, length=burgers_hd.ny-2),
	velocity_magnitude(burgers_hd);xlabel = "x", ylabel = "y", c = color,
    legend=:none
)
end

# ╔═╡ 7ff17e31-f755-4fc5-b66d-64965f92c6d7
begin
set_bc!(burgers)
set_ic!(burgers)
end

# ╔═╡ f152117a-578c-451f-86ce-4acd1a669bfd
dburgers = Enzyme.make_zero(deepcopy(burgers))

# ╔═╡ 826e779b-cc2c-4da1-9430-93dfa4641185
autodiff(ReverseWithPrimal, final_energy!, Active, Duplicated(burgers, dburgers))

# ╔═╡ 6dabc190-b86e-47e7-ae4b-c00a4f83fdd7
function final_energy_chk!(burgers::Burgers, scheme)
    @checkpoint_struct scheme burgers for i = 1:burgers.tsteps
        advance!(burgers)
    end
    return energy(burgers)
end

# ╔═╡ 6989b054-e102-4349-bbb4-f45fabfa4d3e
 revolve = Revolve{Burgers}(tsteps, 2; verbose = 1)

# ╔═╡ 42a682f0-6fc4-44ae-9c3f-1434a69f5ff6
begin
	reset!(revolve)
	autodiff(ReverseWithPrimal, final_energy_chk!, Active, Duplicated(burgers, dburgers), Const(revolve))
end

# ╔═╡ 5beb4b8b-2c08-440a-a275-12fba8bbd852
begin
burgers_long = Burgers(Nx, Ny, μ, dx, dy, dt, 1000)
set_ic!(burgers_long)
set_bc!(burgers_long)
final_energy_chk!(burgers_long, revolve)
end

# ╔═╡ 9c5f3dbe-598c-4160-875f-51de499aba05
surface(
    range(-3, 3, length=burgers_long.nx-2),
    range(-3, 3, length=burgers_long.ny-2),
    velocity_magnitude(burgers_long);xlabel = "x", ylabel = "y", c = color,
    legend=:none
)

# ╔═╡ 389a874b-a855-47eb-a23c-80e2f5e7f027
begin
	set_ic!(burgers_long)
	set_bc!(burgers_long)
	dburgers_long = Enzyme.make_zero(deepcopy(burgers_long))
end

# ╔═╡ f2505bc2-64da-4264-a919-85e58f8d1da1
revolve_long = Revolve{Burgers}(1000, 100; verbose = 1)

# ╔═╡ 7cd9b223-ebc0-47d3-8272-444288f9e9dd
begin
	reset!(revolve_long)
	autodiff(ReverseWithPrimal, final_energy_chk!, Active, Duplicated(burgers_long, dburgers_long), Const(revolve_long))
end

# ╔═╡ eea27917-0307-4a51-b3e6-1f050c46f06a
surface(
    range(-3, 3, length=dburgers_long.nx-2),
    range(-3, 3, length=dburgers_long.ny-2),
    dburgers.lastu[2:end-1, 2:end-1] .^ 2 + dburgers.lastv[2:end-1, 2:end-1] .^ 2;
	xlabel = "x", ylabel = "y", c = color,
    legend=:none
)

# ╔═╡ 7e2ed6b7-d975-4f98-8aa8-d2405440b192
dburgers_long.lastu

# ╔═╡ Cell order:
# ╟─a575949c-2368-11ef-2b2d-2fe3a2ffbae3
# ╠═5748f296-c540-417f-b39d-f82d0e46559e
# ╠═bbe566f7-02b7-480b-bd63-d6c72aa1ac40
# ╠═b96ea551-35ef-4b87-989b-fe5e49e098cc
# ╠═ddf67473-5f5f-44b2-bd40-415772f5aead
# ╟─feb5a4ce-f32e-4f72-94c9-a39460a6866e
# ╠═d7df76fd-dbb3-43bd-a1a9-35b69cebcbcb
# ╟─dd62122a-89c0-471a-97c1-5f3d60b98f40
# ╠═0dc72edd-d11c-4f71-baee-a5c265b2d135
# ╟─75f0cbda-57f4-4232-93f8-a88ccd764d16
# ╠═02e8e500-9785-436c-8e2c-d3455ab8eec3
# ╟─69a5e411-7f1b-4f38-895d-c92a5fb152f7
# ╠═a25e3d03-8eed-4a0b-9fd7-c5e30f9c8f1c
# ╟─eabb6565-946f-4e00-a3ac-9ef33d05c594
# ╠═4b4f81f6-6ff0-430f-8a86-965d4a41044d
# ╟─664315e2-b4db-41ad-82d1-3039454fed1e
# ╠═9c21cfa8-0db3-4d74-9465-75bf25b15a5b
# ╟─12dfa763-b307-4f76-9a0d-5b24e6130da9
# ╠═64b6eca3-8b8a-470c-b496-8207d88fb99c
# ╠═96a54adb-015c-4243-a99d-9cda695f2a4d
# ╠═48b5e47b-e090-401d-ad7a-4898874b5117
# ╠═de82b316-f8b9-479d-acac-dd18768e1e43
# ╠═32aadddf-6220-4e3d-aae1-09d58e173b41
# ╠═c5feb1d9-4703-49a2-bedc-b996887f4d91
# ╠═50c2b95a-9e72-48b9-879b-d31a70c0f6cb
# ╠═bc294229-2b02-4b19-8f1b-71348793b323
# ╠═f9f6b93f-74b4-49bb-91ed-71b14c87fb3a
# ╠═daf57c26-3d7c-48b1-a26f-904f1c39a112
# ╠═f43f9eb8-902a-4489-981d-2c356d025289
# ╟─b8201b3f-5187-4fdd-86a2-feb4e1d4f05b
# ╠═a9814ad4-7c9b-4c25-8fa4-b1683ad9be82
# ╠═7ff17e31-f755-4fc5-b66d-64965f92c6d7
# ╠═06f23f52-0fcc-4707-b06c-80f4529b506d
# ╠═f152117a-578c-451f-86ce-4acd1a669bfd
# ╠═826e779b-cc2c-4da1-9430-93dfa4641185
# ╠═cd9ef5ba-5e59-43ed-82d1-588dc7effefe
# ╠═6dabc190-b86e-47e7-ae4b-c00a4f83fdd7
# ╠═6989b054-e102-4349-bbb4-f45fabfa4d3e
# ╠═42a682f0-6fc4-44ae-9c3f-1434a69f5ff6
# ╠═5beb4b8b-2c08-440a-a275-12fba8bbd852
# ╠═9c5f3dbe-598c-4160-875f-51de499aba05
# ╠═389a874b-a855-47eb-a23c-80e2f5e7f027
# ╠═f2505bc2-64da-4264-a919-85e58f8d1da1
# ╠═7cd9b223-ebc0-47d3-8272-444288f9e9dd
# ╠═eea27917-0307-4a51-b3e6-1f050c46f06a
# ╠═7e2ed6b7-d975-4f98-8aa8-d2405440b192
