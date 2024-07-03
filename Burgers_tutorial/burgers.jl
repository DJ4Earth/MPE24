using CairoMakie
using KernelAbstractions
using Adapt
using Enzyme
using Checkpointing
using CUDA

const KA = KernelAbstractions

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

@kernel function set_ic_kernel!(lastu, nextu, lastv, nextv, nx, ny)
    i, j = @index(Global, NTuple)
    lastu[i+1, j+1] = exp(-(-3.0 + i * 6.0 / (nx - 1))^2 - (-3.0 + j * 6.0 / (ny - 1))^2)
    lastv[i+1, j+1] = exp(-(-3.0 + i * 6.0 / (nx - 1))^2 - (-3.0 + j * 6.0 / (ny - 1))^2)
    nextu[i+1, j+1] = exp(-(-3.0 + i * 6.0 / (nx - 1))^2 - (-3.0 + j * 6.0 / (ny - 1))^2)
    nextv[i+1, j+1] = exp(-(-3.0 + i * 6.0 / (nx - 1))^2 - (-3.0 + j * 6.0 / (ny - 1))^2)
end

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

function energy(burgers::Burgers)
    @inbounds lenergy =
        sum(burgers.nextu[2:end-1, 2:end-1] .^ 2 .+ burgers.nextv[2:end-1, 2:end-1] .^ 2) /
        (burgers.nx*burgers.ny)
end

function advance!(burgers::Burgers)
    stencil!(burgers)
	copyto!(burgers.lastu, burgers.nextu)
    copyto!(burgers.lastv, burgers.nextv)
    return nothing
end

function final_energy!(burgers::Burgers)
    for i = 1:burgers.tsteps
        advance!(burgers)
    end
    return energy(burgers)
end

function final_energy!(burgers::Burgers, scheme)
    @checkpoint_struct scheme burgers for i = 1:burgers.tsteps
        advance!(burgers)
    end
    return energy(burgers)
end

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

function set_ic!(burgers::Burgers)
    set_ic_kernel!(burgers.backend)(
        burgers.lastu, burgers.nextu, burgers.lastv, burgers.nextv, burgers.nx, burgers.ny,
        ndrange=(burgers.nx-2, burgers.ny-2),
    )
    KA.synchronize(burgers.backend)
    return nothing
end

function velocity_magnitude_sq(burgers)
	burgers.nextu[2:end-1, 2:end-1] .^ 2 + burgers.nextv[2:end-1, 2:end-1] .^ 2
end

function adjoint_velocity_magnitude_sq(dburgers)
	dburgers.lastu[2:end-1, 2:end-1] .^ 2 + dburgers.lastv[2:end-1, 2:end-1] .^ 2
end

# function main()
Nx = 1000
Ny = 1000
tsteps = 1000
μ = 0.01 # # U * L / Re,   nu
dx = 1e-2
dy = 1e-2
dt = 1e-3 # dt < 0.5 * dx^2
burgers = Burgers(Nx, Ny, μ, dx, dy, dt, tsteps)
set_ic!(burgers)
set_bc!(burgers)
ienergy = energy(burgers)
fenergy = final_energy!(burgers)
surface(
    range(-3, 3, length=burgers.nx-2),
    range(-3, 3, length=burgers.ny-2),
    velocity_magnitude_sq(burgers);
    axis=(type=Axis3, azimuth = -pi/4,)
)
revolve = Revolve{Burgers}(1000, 10; verbose = 1)
set_bc!(burgers)
set_ic!(burgers)
reset!(revolve)
dburgers = Enzyme.make_zero(deepcopy(burgers))
autodiff(ReverseWithPrimal, final_energy!, Active, Duplicated(burgers, dburgers), Const(revolve))
surface(
    range(-3, 3, length=burgers.nx-2),
    range(-3, 3, length=burgers.ny-2),
	adjoint_velocity_magnitude_sq(dburgers);
	axis=(type=Axis3,),
)

cu_burgers = Burgers(100, 100, μ, 1e-2, 1e-2, dt, 2, CUDABackend())
set_ic!(cu_burgers)
set_bc!(cu_burgers)
final_energy!(cu_burgers)

burgers = Burgers(100, 100, μ, 1e-2, 1e-2, dt, 2, CPU())
set_ic!(burgers)
set_bc!(burgers)
final_energy!(burgers)

set_bc!(burgers)
set_ic!(burgers)
dburgers = Enzyme.make_zero(deepcopy(burgers))
autodiff(ReverseWithPrimal, final_energy!, Active, Duplicated(burgers, dburgers))

# Fails due to Active kernel arguments
# set_bc!(cu_burgers)
# set_ic!(cu_burgers)
# cu_dburgers = Enzyme.make_zero(deepcopy(cu_burgers))
# autodiff(ReverseWithPrimal, final_energy!, Active, Duplicated(cu_burgers, cu_dburgers))