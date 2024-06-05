using Adapt
using CUDA
using Enzyme
using LinearAlgebra
using Checkpointing
using KernelAbstractions
const KA = KernelAbstractions
using Plots

mutable struct Burgers
    nextu::AbstractMatrix
    nextv::AbstractMatrix
    lastu::AbstractMatrix
    lastv::AbstractMatrix
    nx::Int
    ny::Int
    μ::Float64
    dx::Float64
    dy::Float64
    dt::Float64
    tsteps::Int
    backend::KA.Backend
end

function energy(burgers::Burgers)
    @inbounds lenergy =
        sum(burgers.nextu[2:end-1, 2:end-1] .^ 2 .+ burgers.nextv[2:end-1, 2:end-1] .^ 2) /
        (burgers.nx*burgers.ny)
end

function final_energy(burgers::Burgers)
    for i = 1:burgers.tsteps
        advance!(burgers)
    end
    return energy(burgers)
end

function final_energy_chk!(burgers::Burgers, scheme)
    @checkpoint_struct scheme burgers for i = 1:burgers.tsteps
        advance!(burgers)
    end
    return energy(burgers)
end

function advance!(burgers::Burgers)
    stencil!(burgers)
    # set_bc!(burgers)
    (tmpu, tmpv) = (burgers.lastu, burgers.lastv)
    (burgers.lastu, burgers.lastv) = (burgers.nextu, burgers.nextv)
    (burgers.nextu, burgers.nextv) = (tmpu, tmpv)
    return nothing
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

@kernel function set_ic_kernel!(lastu, nextu, lastv, nextv, nx, ny)
    i, j = @index(Global, NTuple)
    lastu[i+1, j+1] = exp(-(-3.0 + i * 6.0 / (nx - 1))^2 - (-3.0 + j * 6.0 / (ny - 1))^2)
    lastv[i+1, j+1] = exp(-(-3.0 + i * 6.0 / (nx - 1))^2 - (-3.0 + j * 6.0 / (ny - 1))^2)
    nextu[i+1, j+1] = exp(-(-3.0 + i * 6.0 / (nx - 1))^2 - (-3.0 + j * 6.0 / (ny - 1))^2)
    nextv[i+1, j+1] = exp(-(-3.0 + i * 6.0 / (nx - 1))^2 - (-3.0 + j * 6.0 / (ny - 1))^2)
end


function set_ic!(burgers::Burgers)
    set_ic_kernel!(burgers.backend)(
        burgers.lastu, burgers.nextu, burgers.lastv, burgers.nextv, burgers.nx, burgers.ny,
        ndrange=(burgers.nx-2, burgers.ny-2),
    )
    KA.synchronize(burgers.backend)
    return nothing
end

# Create object from struct.
function burgers(
    Nx::Int64,
    Ny::Int64,
    tsteps::Int64,
    μ::Float64,
    dx::Float64,
    dy::Float64,
    dt::Float64,
    backend=CPU()
)
    burgers = Burgers(Nx, Ny, μ, dx, dy, dt, tsteps, backend)

    # Boundary conditions
    set_ic!(burgers)
    set_bc!(burgers)

    vel = (burgers.nextu[2:end-1, 2:end-1] .^ 2 + burgers.nextv[2:end-1, 2:end-1] .^ 2)
    # heatmap(vel)
    ienergy = energy(burgers)

    println("Initial energy E = $ienergy")
    set_bc!(burgers)
    set_ic!(burgers)
    fenergy = final_energy(burgers)
    println("Final energy E = $fenergy")

    vel = (burgers.nextu[2:end-1, 2:end-1] .^ 2 + burgers.nextv[2:end-1, 2:end-1] .^ 2)
    # display(heatmap(vel))
    return ienergy, fenergy, burgers
end

function burgers_adjoint(
    Nx::Int64,
    Ny::Int64,
    tsteps::Int64,
    μ::Float64,
    dx::Float64,
    dy::Float64,
    dt::Float64,
    snaps::Int64;
    storage = ArrayStorage{Burgers}(snaps),
)
    burgers = Burgers(Nx, Ny, μ, dx, dy, dt, tsteps)
    set_bc!(burgers)
    set_ic!(burgers)
    revolve = Revolve{Burgers}(tsteps, snaps; verbose = 1, storage = storage)
    # revolve = Periodic{Burgers}(tsteps, snaps; verbose=1, storage=storage)

    @time begin
        set_bc!(burgers)
        set_ic!(burgers)
        dburgers = Enzyme.make_zero(deepcopy(burgers))
        ret = autodiff(ReverseWithPrimal, final_energy, Active, Duplicated(burgers, dburgers))
        # ret = autodiff(ReverseWithPrimal, final_energy_chk!, Active, Duplicated(burgers, dburgers), Const(revolve))
        @show ret
    end

    dvel = (dburgers.lastu .^ 2 + dburgers.lastv .^ 2)
    println("Norm of energy with respect to initial velocity norm(dE/dv0) = $(norm(dvel))")
    # heatmap(dvel)
    # heatmap(dburgers[1].lastu[2:end-1,2:end-1])
    return norm(dvel), ret[2], burgers, dburgers
end

function main(backend)
    scaling = 1

    Nx = 100 * scaling
    Ny = 100 * scaling
    tsteps = 1000 * scaling

    μ = 0.01 # # U * L / Re,   nu

    dx = 1e-1
    dy = 1e-1
    dt = 1e-3 # dt < 0.5 * dx^2

    snaps = 100
    println(
        "Running Burgers with Nx = $Nx, Ny = $Ny, tsteps = $tsteps, μ = $μ, dx = $dx, dy = $dy, dt = $dt, snaps = $snaps",
    )
    ienergy, fenergy, fburgers = burgers(Nx, Ny, tsteps, μ, dx, dy, dt, backend)
    dlastu = Float64[]
    for h in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
        hburgers = Burgers(Nx, Ny, μ, dx, dy, dt, tsteps)
        set_ic!(hburgers)
        set_bc!(hburgers)
        hburgers.lastu[55, 46] += h
        push!(dlastu, (final_energy(hburgers) - fenergy) / h)
    end
    ndvel, adenergy, adburgers, dburgers =
        burgers_adjoint(Nx, Ny, tsteps, μ, dx, dy, dt, snaps)
    println("Primal (f, adf): $fenergy $adenergy")
    isapprox(fenergy, adenergy)
    isapprox(ienergy, adenergy)
    # @show ienergy ≈ 0.0855298595153226
    # @show fenergy ≈ 0.08426001732938161
    # @show ndvel ≈ 1.3020729832060115e-6
    @show isapprox(dlastu[2], dburgers.lastu[55, 46], atol = 1e-4)
    @show dlastu
    @show dburgers.lastu[55, 46]
    dburgers.lastu[55, 46] - dlastu[2]
    return adburgers, dburgers
    # return fburgers, fburgers
end

fburgers, dburgers = main(CPU());
vel = (fburgers.nextu[2:end-1, 2:end-1] .^ 2 + fburgers.nextv[2:end-1, 2:end-1] .^ 2)

dvel = (dburgers.lastu[2:end-1, 2:end-1] .^ 2 + dburgers.lastv[2:end-1, 2:end-1] .^ 2)
heatmap(adapt(CPU(), vel))
heatmap(adapt(CPU(), dvel))
