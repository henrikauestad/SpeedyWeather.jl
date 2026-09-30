@testset "RayleighSponge profile" begin
    spectral_grid = SpectralGrid(truncation = 21, nlayers = 16)
    sponge = RayleighSponge(spectral_grid, sigma = 0.3, time_scale = Hour(12), time_scale_div = Day(2))
    model = PrimitiveDryModel(spectral_grid, forcing = sponge)
    initialize!(sponge, model)

    σ = Array(model.geometry.σ_levels_full)
    rate, rate_div = Array(sponge.rate), Array(sponge.rate_div)

    # no damping below the sponge, damping above
    @test all(rate[σ .>= 0.3] .== 0)
    @test all(rate[σ .< 0.3] .> 0)

    # increasing towards the top (k = 1), bounded by 1/time_scale
    in_sponge = σ .< 0.3
    @test issorted(rate[in_sponge], rev = true)
    @test maximum(rate) < 1 / (12 * 3600)
    @test rate ≈ rate_div .* 4    # 12 hours vs 2 days

    # switches
    sponge_div_only = RayleighSponge(spectral_grid, damp_vorticity = false)
    initialize!(sponge_div_only, model)
    @test all(sponge_div_only.rate .== 0)
    @test any(sponge_div_only.rate_div .> 0)
end

@testset "RayleighSponge tendencies" begin
    spectral_grid = SpectralGrid(truncation = 21, nlayers = 8)
    NF = spectral_grid.NF

    for damp_zonal_mean in (false, true)
        sponge = RayleighSponge(spectral_grid; damp_zonal_mean)
        model = PrimitiveDryModel(spectral_grid, forcing = sponge)
        simulation = initialize!(model)
        vars = simulation.variables
        TS = model.time_stepping

        # random state in both leapfrog steps, zero tendencies
        vars.prognostic.vorticity .= rand(Complex{NF}, size(vars.prognostic.vorticity)...)
        vars.prognostic.divergence .= rand(Complex{NF}, size(vars.prognostic.divergence)...)
        vars.tendencies.vorticity .= 0
        vars.tendencies.divergence .= 0

        SpeedyWeather.forcing!(vars, model)

        # the sponge reads the previous (1st) leapfrog step
        vor = Array(SpeedyWeather.get_step(vars.prognostic.vorticity, 1).data)
        div = Array(SpeedyWeather.get_step(vars.prognostic.divergence, 1).data)
        vor_tend = Array(SpeedyWeather.get_tendency_step(vars.tendencies.vorticity, TS, sponge).data)
        div_tend = Array(SpeedyWeather.get_tendency_step(vars.tendencies.divergence, TS, sponge).data)
        m = Array(vars.prognostic.vorticity.spectrum.m_indices)
        rate, rate_div = Array(sponge.rate), Array(sponge.rate_div)

        damped = damp_zonal_mean ? trues(length(m)) : m .> 1
        @test vor_tend ≈ -damped .* rate' .* vor
        @test div_tend ≈ -damped .* rate_div' .* div
        @test all(iszero, vor_tend[.!damped, :])        # zonal mean untouched unless damp_zonal_mean
        @test all(iszero, vor_tend[:, end])             # layers below the sponge are untouched
        @test any(!iszero, vor_tend[damped, 1])         # top layer is damped
    end
end

@testset "RayleighSponge in simulations" begin
    # combined with Held-Suarez via a NamedTuple of forcings
    # 48 equally spaced layers need a shorter time step (thin top layers), independent of the sponge
    for (Model, trunc, nlayers, Δt_at_T32, period) in (
            (PrimitiveDryModel, 32, 8, Minute(40), Day(2)),      # T31L8
            (PrimitiveDryModel, 128, 48, Minute(10), Day(1)),    # T127L48
            (PrimitiveWetModel, 24, 8, Minute(40), Day(2)),      # T23L8, coarse wet
        )
        spectral_grid = SpectralGrid(truncation = trunc; nlayers)
        forcing = (held_suarez = HeldSuarez(spectral_grid), sponge = RayleighSponge(spectral_grid))
        drag = LinearDrag(spectral_grid)
        time_stepping = Leapfrog(spectral_grid; Δt_at_T32)
        model = Model(spectral_grid; forcing, drag, time_stepping)
        simulation = initialize!(model)
        run!(simulation; period)
        @test simulation.model.feedback.nans_detected == false
    end

    # sponge alone
    spectral_grid = SpectralGrid(truncation = 21, nlayers = 8)
    model = PrimitiveWetModel(spectral_grid, forcing = RayleighSponge(spectral_grid))
    simulation = initialize!(model)
    run!(simulation, period = Day(2))
    @test simulation.model.feedback.nans_detected == false
end
