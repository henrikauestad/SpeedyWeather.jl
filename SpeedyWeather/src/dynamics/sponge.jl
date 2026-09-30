export RayleighSponge

"""
Rayleigh sponge layer at the top of the model domain for the primitive equation models.
Linearly damps vorticity and divergence in spectral space at all wavenumbers with a rate that
increases from 0 at σ = `sigma` to 1/`time_scale` (or 1/`time_scale_div`) at the model top
σ = 0 following a sin² profile

    rate(σ) = sin²(π/2 * (sigma - σ) / sigma) / time_scale  for σ < sigma, 0 otherwise.

Damping vorticity and divergence at the same rate is Rayleigh friction on (u, v). By default the
zonal mean (order m = 0) is not damped, so the sponge damps the waves (eddies) but not the
zonal-mean flow. This avoids the spurious downward influence of a zonal-mean sponge that
violates angular momentum conservation (Shepherd et al. 1996, JGR; Shepherd and Shaw 2004, JAS).
The damping is evaluated at the previous leapfrog step (forward in time over 2Δt) as a damping
term evaluated at the centred step is unstable with leapfrog. Stable for time scales > Δt.
Fields are
$(TYPEDFIELDS)"""
@parameterized @kwdef struct RayleighSponge{NF, VectorType} <: AbstractForcing
    "[OPTION] σ level where the sponge starts, no damping for σ ≥ sigma"
    @param sigma::NF = 0.2 (bounds = 0 .. 1,)

    "[OPTION] damping time scale for vorticity at the model top σ = 0"
    time_scale::Second = Day(1)

    "[OPTION] damping time scale for divergence at the model top σ = 0"
    time_scale_div::Second = Day(1)

    "[OPTION] damp vorticity?"
    damp_vorticity::Bool = true

    "[OPTION] damp divergence?"
    damp_divergence::Bool = true

    "[OPTION] also damp the zonal mean (m = 0)? Violates angular momentum conservation"
    damp_zonal_mean::Bool = false

    "[DERIVED] damping rate for vorticity per layer [1/s]"
    rate::VectorType

    "[DERIVED] damping rate for divergence per layer [1/s]"
    rate_div::VectorType
end

"""$(TYPEDSIGNATURES)
Create a `RayleighSponge` with the damping rate arrays allocated given `spectral_grid`."""
function RayleighSponge(SG::SpectralGrid; kwargs...)
    rate = on_architecture(SG.architecture, zeros(SG.NF, SG.nlayers))
    rate_div = on_architecture(SG.architecture, zeros(SG.NF, SG.nlayers))
    return RayleighSponge{SG.NF, SG.VectorType}(; rate, rate_div, kwargs...)
end

"""$(TYPEDSIGNATURES)
Vertical profile of the sponge in [0, 1], 0 for `σ ≥ sigma` increasing to 1 at `σ = 0`."""
sponge_profile(σ, sigma) = σ < sigma ? sinpi((sigma - σ) / (2 * sigma))^2 : zero(σ)

"""$(TYPEDSIGNATURES)
Precompute the damping rates per layer [1/s] of the `RayleighSponge`."""
function initialize!(sponge::RayleighSponge, model::PrimitiveEquation)
    (; sigma, damp_vorticity, damp_divergence, rate, rate_div) = sponge
    σ = model.geometry.σ_levels_full
    NF = eltype(rate)

    # inverse time scales [1/s], 0 if switched off
    r_vor = damp_vorticity ? NF(1 / Second(sponge.time_scale).value) : zero(NF)
    r_div = damp_divergence ? NF(1 / Second(sponge.time_scale_div).value) : zero(NF)

    rate .= r_vor .* sponge_profile.(σ, sigma)
    rate_div .= r_div .* sponge_profile.(σ, sigma)
    return nothing
end

"""$(TYPEDSIGNATURES)
Damp vorticity and divergence in the sponge layer towards zero, skipping the
zonal mean (m = 0) unless `sponge.damp_zonal_mean`."""
function forcing!(vars::Variables, sponge::RayleighSponge, model::PrimitiveEquation)
    (; time_stepping) = model
    vor = get_prognostic_step(vars.prognostic.vorticity, time_stepping, sponge)
    div = get_prognostic_step(vars.prognostic.divergence, time_stepping, sponge)
    vor_tend = get_tendency_step(vars.tendencies.vorticity, time_stepping, sponge)
    div_tend = get_tendency_step(vars.tendencies.divergence, time_stepping, sponge)

    # vor, div are scaled by radius, as are their tendencies in scale_tendencies! after forcing!
    # so -rate*vor here is the correct tendency in the radius-scaled equations
    (; rate, rate_div, damp_zonal_mean) = sponge
    launch!(
        architecture(vor_tend), SpectralWorkOrder, size(vor_tend), rayleigh_sponge_kernel!,
        vor_tend, div_tend, vor, div, rate, rate_div, vor.spectrum.m_indices, damp_zonal_mean
    )
    return nothing
end

@kernel inbounds = true function rayleigh_sponge_kernel!(
        vor_tend, div_tend, vor, div, rate, rate_div, m_indices, damp_zonal_mean
    )
    lm, k = @index(Global, NTuple)

    # m_indices are 1-based so m = 0 (zonal mean) is m_indices[lm] == 1
    if damp_zonal_mean || m_indices[lm] > 1
        vor_tend[lm, k] -= rate[k] * vor[lm, k]
        div_tend[lm, k] -= rate_div[k] * div[lm, k]
    end
end
