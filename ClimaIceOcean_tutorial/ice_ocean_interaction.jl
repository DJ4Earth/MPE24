using Oceananigans.Operators

using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models: AbstractModel
using Oceananigans.TimeSteppers: tick!
using Oceananigans.Utils: launch!

using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll

# Simulations interface
import Oceananigans: fields, prognostic_fields
import Oceananigans.Fields: set!
import Oceananigans.Architectures: AbstractArchitecture
import Oceananigans.Grids: architecture
import Oceananigans.Models: timestepper, NaNChecker
import Oceananigans.OutputWriters: default_included_properties
import Oceananigans.Simulations: reset!, initialize!, iteration
import Oceananigans.TimeSteppers: time_step!, update_state!, time
import Oceananigans.Utils: prettytime

function ice_ocean_latent_heat!(ice_model, ocean_model, Δt)
    ρₒ = 1024 # ocean_density
    cₒ = 3991 # ocean_heat_capacity
    Qₒ = ice_model.external_heat_fluxes.bottom
    Tₒ = ocean_model.tracers.T
    Sₒ = 30 #ocean.model.tracers.S .     NOOOOOO YOU NEED SALT!
    hᵢ = ice_model.ice_thickness

    liquidus = ice_model.phase_transitions.liquidus
    grid = ocean_model.grid
    arch = architecture(grid)

    launch!(arch, grid, :xy, _compute_ice_ocean_latent_heat!,
            Qₒ, grid, hᵢ, Tₒ, Sₒ, liquidus, ρₒ, cₒ, Δt)

    return nothing
end

@kernel function _compute_ice_ocean_latent_heat!(latent_heat,
                                                 grid,
                                                 ice_thickness,
                                                 ocean_temperature,
                                                 ocean_salinity,
                                                 liquidus,
                                                 ρₒ, cₒ, Δt)

    i, j = @index(Global, NTuple)

    Nz = size(grid, 3)
    Qₒ = latent_heat
    hᵢ = ice_thickness
    Tₒ = ocean_temperature
    Sₒ = ocean_salinity

    δQ = zero(grid)
    icy_cell = @inbounds hᵢ[i, j, 1] > 0 # make ice bath approximation then

    @unroll for k = Nz:-1:1
        @inbounds begin
            # Various quantities
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            Tᴺ = Tₒ[i, j, k]
            Sᴺ = Sₒ[i, j, k]
        end

        # Melting / freezing temperature at the surface of the ocean
        Tₘ = melting_temperature(liquidus, Sᴺ)
                                 
        # Conditions for non-zero ice-ocean flux:
        #   - the ocean is below the freezing temperature, causing formation of ice.
        freezing = Tᴺ < Tₘ 

        #   - We are at the surface and the cell is covered by ice.
        icy_surface_cell = (k == Nz) & icy_cell

        # When there is a non-zero ice-ocean flux, we will instantaneously adjust the
        # temperature of the grid cells accordingly.
        adjust_temperature = freezing | icy_surface_cell

        # Compute change in ocean heat energy.
        #
        #   - When Tᴺ < Tₘ, we heat the ocean back to melting temperature by extracting heat from the ice,
        #     assuming that the heat flux (which is carried by nascent ice crystals called frazil ice) floats
        #     instantaneously to the surface.
        #
        #   - When Tᴺ > Tₘ and we are in a surface cell covered by ice, we assume equilibrium
        #     and cool the ocean by injecting excess heat into the ice.
        # 
        δEₒ = adjust_temperature * ρₒ * cₒ * (Tₘ - Tᴺ)

        # Perform temperature adjustment
        @inline Tₒ[i, j, k] = ifelse(adjust_temperature, Tₘ, Tᴺ)

        # Compute the heat flux from ocean into ice.
        #
        # A positive value δQ > 0 implies that the ocean is cooled; ie heat
        # is fluxing upwards, into the ice. This occurs when applying the
        # ice bath equilibrium condition to cool down a warm ocean (δEₒ < 0).
        #
        # A negative value δQ < 0 implies that heat is fluxed from the ice into
        # the ocean, cooling the ice and heating the ocean (δEₒ > 0). This occurs when
        # frazil ice is formed within the ocean.
        
        δQ -= δEₒ * Δz / Δt
    end

    # Store ice-ocean flux
    @inbounds Qₒ[i, j, 1] = δQ
end
