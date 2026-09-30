# Rayleigh sponge layer at the model top as an `AbstractForcing`

> Status: **in progress**. `RayleighSponge` is implemented as a forcing with unit tests; the first reflection experiment (T31L30) is inconclusive because Held–Suarez does not produce vertically propagating planetary waves.

Date of initial draft: 2026-09-30

Base revision: `c7d631c658feefc851eb7af5b93aa5754430faf5` (`main`, branch `sponge`)

## Originating prompt

> Let's plan the project. I would like to add a sponge layer (Rayleight sponge) to the top of
> the model domain. It should be implemented as an AbstractForcing. The model already has hyper
> diffusion, but at the top of the model domain, all wave numbers need to be dampened to prevent
> wave reflecting on the top boundary. It is unclear to me whether a classic Rayleigh dampening
> affecting both divergence and vorticity in spectral space is better than only dampening
> diveregen (vertical velocity) and then adding dumerical diffusion for the other wave numbers.
>
> https://journals.ametsoc.org/view/journals/atsc/61/23/jas-3295.1.xml
> https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/96JD01994
> https://clima.github.io/ClimaAtmos.jl/dev/sponge/

## Revision log

- **2026-09-30, initial draft.**
- **2026-09-30.** The user asked how hyperdiffusion is implemented, then: "Please also consider
  the option of implementing the sponge as a callback". Added *Alternatives considered*, which
  compares the forcing with a diffusion-slot variant and a callback variant.
- **2026-09-30.** The user chose: "Write it as a forcing, please. Do the test described, and plot the
  output", then "please run the tests also with T127N48 (primitivedrymodel) as well as a courser
  primitivewetmodel". Implemented route A with the recommended defaults (eddies only,
  `sigma = 0.2`, 1 day, no temperature damping). While implementing:
  - `which_prognostic_step` for the sponge lives in `leapfrog.jl`, because `AbstractLeapfrog`
    is defined after `forcing.jl`.
  - The simulation tests cover T31L8 dry, T127L48 dry and T23L8 wet. The new convention is
    `truncation = T + 1`, so T127 is `truncation = 128`.
  - 48 equally spaced layers are unstable with the default time step at T31, T63 and T127,
    also without any forcing (a pre-existing issue from the thin top layers). T127L48 needs
    `Δt_at_T32 = Minute(10)`, i.e. Δt = 150 s.
  - Reflection experiment: `../experiments/reflection.jl` (outside the repo). T31, 30
    log-pressure spaced layers up to σ = 0.002, Held–Suarez plus a 3 km wave-2 mountain
    at 25–65˚N, sponge from σ = 0.02 with a 1-day time scale, Δt = 20 min, 400 days with
    100 days of spin-up. The results are in the *Results* section.

## Problem description

The model lid of the primitive equation models is at σ = 0. Upward-propagating waves reach
the top layers, reflect off the lid and travel back down. Real waves are not reflected there.
The reflected waves change the stratospheric and tropospheric circulation. To stop this, a
sponge layer is needed that absorbs waves in the top few layers before they reach the lid.

The existing `HyperDiffusion` is strongly scale-selective (∇⁸ by default), so it barely
touches the large scales. Above `tapering_σ = 0.2` it reduces the power for **divergence
only**, down to `power_stratosphere = 2` (∇⁴). That is already a divergence-only "sponge",
but it is still scale-selective. Planetary-scale waves (low wavenumbers) are not damped, and
vorticity is not damped aloft at all beyond the ∇⁸ diffusion.

## Background

### The references

1. **Shepherd, Semeniuk & Koshyk (1996)**, *Sponge layer feedbacks in middle-atmosphere
   models*, JGR 101(D18), 23447–23464, doi:10.1029/96JD01994.
   Because the sponge is a relaxation, it couples artificially to the dynamics below it. A
   Rayleigh sponge acting on the **zonal-mean** flow turns forcing or heating below into a
   drag in the sponge. That diverts part of the mean meridional circulation upward and changes
   temperatures well below the sponge. The effect decays roughly like exp(−Δz/H) below the
   sponge, but it does not vanish.
2. **Shepherd & Shaw (2004)**, *The angular momentum constraint on climate sensitivity and
   downward influence in the middle atmosphere*, JAS 61, 2899–2908, doi:10.1175/JAS-3295.1.
   To respect angular momentum conservation and avoid spurious downward influence, a model
   must have *no zonal-mean sponge layer*. The sponge may damp waves (eddies) but not the
   zonal-mean wind.
3. **ClimaAtmos.jl sponge docs.** Two sponges share one profile,
   β(z) = sin²(π/2 · (z − z_d)/(z_top − z_d)) for z > z_d, else 0.
   - A *Rayleigh sponge* by default damps **only the vertical velocity w**
     (α_w = 1 s⁻¹); damping the horizontal velocity is optional and off by default.
   - A *viscous sponge* is ∇² diffusion with coefficient κ₂·β(z), applied to velocities,
     energy, water and tracers.

   ClimaAtmos is non-hydrostatic and has w as a prognostic variable. SpeedyWeather is
   hydrostatic: σ̇ (the vertical velocity) is diagnosed from the vertically integrated
   divergence, so in our model "damp w" means "damp divergence".

### Which waves reflect in SpeedyWeather?

- **Gravity waves** are mostly divergent. At T31 with 8 layers they are poorly resolved and
  already heavily damped aloft by the divergence hyperdiffusion (∇⁴, 1 h time scale).
- **Vertically propagating planetary (Rossby) waves** are quasi-geostrophic, so they are mostly
  rotational: vorticity plus the temperature they are balanced with. They have zonal
  wavenumbers m = 1–3 and are the main reflection problem for stratosphere–troposphere studies.
  At T31, with `power = 4`, their vorticity hyperdiffusion rate at l = 3 is about
  (12/930)⁴ ≈ 3·10⁻⁸ of the rate at the truncation, so they are effectively undamped. Even the
  ∇⁴ divergence diffusion aloft only reaches a time scale of about 2 months at l = 3.

### Design decision: vorticity + divergence, eddies only, one component

My recommendation is a single `RayleighSponge <: AbstractForcing` that damps **both vorticity
and divergence**, at all wavenumbers, **excluding the zonal mean (m = 0)** by default.

- **Divergence only plus diffusion is mostly what `HyperDiffusion` already does.** It would
  add little beyond a scale-independent divergence damping, and it does not absorb Rossby
  waves, which are rotational.
- **Damping vorticity is needed to absorb planetary waves.** Damping vorticity and divergence
  at the same rate r is exactly Rayleigh friction on (u, v). This is the "classic" sponge in
  spectral form (e.g. Polvani & Kushner 2002 apply it to u, v in grid space).
- **Keeping m = 0 undamped satisfies Shepherd et al. (1996, 2004).** The zonal-mean zonal wind
  is carried entirely by the m = 0 vorticity coefficients, and the zonal-mean meridional
  circulation v̄ by the m = 0 divergence coefficients. Skipping m = 0 is exact and free in
  spectral space: no zonal mean has to be computed in grid space.
- **Both options become configurations of one component.** Separate switches and time scales
  for vorticity and divergence make "divergence only" (`damp_vorticity = false`) and "full
  Rayleigh including the zonal mean" (`damp_zonal_mean = true`) one keyword away. We can then
  settle the open question with experiments (see *Testing and verification*) instead of in
  advance.

### Alternatives considered: forcing, diffusion slot or callback

**Order of one time step** (`time_step!` in `time_stepping/time_integration.jl`, Leapfrog with
the semi-implicit scheme):

```
reset_tendencies!                 # all tendencies → 0
parameterization_tendencies!      # physics (grid space)
dynamics_tendencies!              # forcing!, drag!  → then the dynamical core, transform to spectral
diffusion_and_implicit!           # implicit_correction!, then horizontal_diffusion!
update_prognostic!                # leapfrog + Robert/Williams filter
transform!                        # new spectral state → grid variables used by the next step
output!
callback!(model.callbacks, …)     # ← callbacks run here, after everything
```

**A. `AbstractForcing` (proposed).** It adds −r·ψ to the spectral tendency inside
`dynamics_tendencies!`, so it runs before the tendency is complete.

- It must be explicit: evaluated at the lagged leapfrog step, stable for τ > Δt.
- It is a normal tendency term, so it appears in the tendencies. It works with the semi-implicit
  solver and the Robert/Williams filter like any other forcing, and it is `@parameterized`.
- Skipping m = 0 is trivial.
- It needs the NamedTuple `forcing!` method so it can be combined with `HeldSuarez`.

**B. Diffusion slot (a subtype of `AbstractHorizontalDiffusion`, or an extension of
`HyperDiffusion`).** A Rayleigh sponge is diffusion with power 0: its rate does not depend on l.
Diffusion runs once the tendency is complete, so it can be implicit via the `expl`/`impl`
tables (`tend = (tend + expl·ψ)·impl`), which is unconditionally stable.

- The tables are indexed by degree l only. Eddies-only needs a per-m mask in the kernel.
- It occupies the slot of `HyperDiffusion`, so it would have to wrap or extend it rather than
  sit next to it. That couples two separate concerns: scale-selective numerical diffusion and a
  physical sponge.
- This is the fallback if time scales close to Δt are ever needed.

**C. Callback (`AbstractCallback`).** `callback!(callback, vars, model)` is called after the
full time step. The docs (`docs/src/callbacks.md`) describe callbacks as mainly diagnostic, and
any tendency written there is zeroed by `reset_tendencies!` before it is used. A callback
sponge would therefore have to be an **operator-split filter on the prognostic state**: it
multiplies the spectral vorticity and divergence of the newest leapfrog level, for m > 0 in
the sponge layers, by an exact damping factor `exp(−2rΔt)`. Since ψⁿ⁺¹ = ψⁿ⁻¹ + 2Δt·F, this
damps both leapfrog chains at the rate r.

- Pros:
  - Unconditionally stable and exact for any time scale.
  - Needs **no changes to the package core**: no NamedTuple forcing, and it can live in a
    user script.
  - Combines trivially with `HeldSuarez`, since `model.callbacks` is a dictionary.
  - Can be added or removed during a run with `add!`/`delete!`, and scheduled with `Schedule`.
- Cons:
  - **The grid state becomes stale.** `transform!` has already turned the undamped spectral
    state into the grid u, v, ζ, D, and the next step computes its nonlinear tendencies from
    those grid fields. Keeping them consistent means transforming vorticity and divergence
    again (at least for the sponge layers) inside the callback. That costs extra and depends
    on the `transform!` internals.
  - `output!` runs before `callback!`, so the output shows the state one filter application
    behind.
  - The damping bypasses the tendencies and the Robert/Williams filter, and the docs warn that
    editing spectral prognostic variables directly "can easily lead to stability issues".
  - It is not a model component in the forcing/drag sense: it is not `@parameterized`, is not
    shown with the model, and callbacks sit outside the Reactant/Enzyme-traced time step.

**Recommendation.** Keep **A** as the implementation. Sponge time scales of hours to days are
well inside the explicit stability limit (2Δt/τ ≈ 0.06 for τ = 1 day at T31). **C** is still
useful as a cheap *prototype*: a script-level callback could test the reflection experiment
before any package code changes. It is not a good final design because of the stale grid
state. Keep **B** for very short time scales.

## Summary of changes

### New file `SpeedyWeather/src/dynamics/sponge.jl` (included after `forcing.jl`)

```julia
export RayleighSponge

"""Rayleigh sponge layer at the model top. Damps vorticity and divergence linearly
at all wavenumbers with a rate that increases from 0 at σ = `sigma` to 1/`time_scale`
at σ = 0 following a sin² profile. The zonal mean (m = 0) is not damped by default
(Shepherd et al. 1996, Shepherd & Shaw 2004).
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
```

- `RayleighSponge(spectral_grid; kwargs...)` allocates `rate` and `rate_div` (`nlayers`) on
  the architecture.
- `initialize!(sponge, model::PrimitiveEquation)` computes
  `β_k = sin²(π/2 · (sigma − σ_k)/sigma)` for σ_k < sigma (else 0) at `σ_levels_full`, then
  sets `rate = β / time_scale` (0 if switched off), the same for `rate_div`.
  - The profile is in σ ≈ p/pₛ, matching `HyperDiffusion`'s `tapering_σ` convention.
  - `initialize!` is defined only for `PrimitiveEquation`, like `HeldSuarez`. The 2D models
    have no vertical structure to sponge.
- `forcing!(vars, sponge::RayleighSponge, model)`:
  - reads the prognostic `vorticity` and `divergence` at the **previous** leapfrog step:
    `which_prognostic_step(var, ::AbstractLeapfrog, ::RayleighSponge) = 1`;
  - launches one `SpectralWorkOrder` kernel that does
    `vor_tend[lm, k] -= rate[k] * vor[lm, k]` (and the same for divergence), skipping
    coefficients with `m_indices[lm] == 1` (1-based m = 0) unless `damp_zonal_mean`.
- **Radius scaling.** The prognostic ζ is stored scaled by the radius R, and
  `scale_tendencies!` multiplies all tendencies by R after `forcing!`. Adding `−r·(Rζ)` before
  that scaling therefore gives d(Rζ)/dt′ = −R·r·(Rζ), which is correct in the scaled time
  t′ = t/R. `StochasticStirring` already writes to `vars.tendencies.vorticity` this way. A unit
  test checks that the term survives `transform!` and `spectral_tendencies!` (both accumulate
  with `add = true`).

**Why the lagged time step.** A damping term evaluated at the *centred* leapfrog step
(which is what forcings do by default: step 2) is unconditionally unstable for the
computational mode. The Robert filter only hides this for weak damping. Evaluating the term
at step 1 is forward-in-time over 2Δt, which is stable for τ > Δt. At T31 (Δt = 40 min) a
1-day sponge has 2Δt/τ ≈ 0.06, well inside the accurate range. `HyperDiffusion` also reads
step 1.

### Combining forcings: `forcing!` for a `NamedTuple`

`model.forcing` holds one component, and the Held–Suarez setup already uses it for
`HeldSuarez`. `initialize!(::NamedTuple, model)` and `variables(::NamedTuple, model)` already
exist (they are used for `greenhouse_gases`), so the only missing piece is in `forcing.jl`:

```julia
forcing!(vars, forcings::NamedTuple, model) = foreach(f -> forcing!(vars, f, model), values(forcings))
```

(written as an unrolled recursion if Enzyme or JET need it, like `_reset_tendencies_inner!`).
Then:

```julia
forcing = (held_suarez = HeldSuarez(spectral_grid), sponge = RayleighSponge(spectral_grid))
model = PrimitiveDryModel(spectral_grid; forcing, drag = LinearDrag(spectral_grid), ...)
```

We need to check that `@parameterized`/`parameters(model)` and the model `show` handle a
NamedTuple in the `forcing` slot. If they don't, fall back to a small `ForcingTuple` wrapper
component.

## Testing and verification

**Unit tests**, new file `SpeedyWeather/test/dynamics/sponge.jl`, run with
`--check-bounds=yes`:

1. Profile: `rate` is 0 for σ ≥ `sigma`, positive and strictly increasing towards the top,
   with 1/`time_scale` as the upper bound; the switches zero `rate`/`rate_div`.
2. Tendency: with random prognostic vorticity and divergence and zeroed tendencies,
   `forcing!` gives `tend == −rate[k] · var` for m > 0, 0 for m = 0, and 0 in layers below
   the sponge. With `damp_zonal_mean = true` the m = 0 coefficients are damped too.
3. It reads the previous leapfrog step, not the current one.
4. Integration: `PrimitiveDryModel` with `forcing = (held_suarez = …, sponge = …)` runs for a
   few days at T31, 8 layers without NaN, and the `forcing = RayleighSponge(...)` form also
   works. `PrimitiveWetModel` gets a smoke test.
5. Existing `forcing_drag.jl` and `dispatch.jl` (JET) tests still pass.

**GPU:** run the unit tests on Metal locally, with Float32. `m_indices` from the `Spectrum`
is already on the device.

**Scientific verification** (a separate script outside the test suite; results in
`notes.md`):

- **Stationary planetary wave reflection test.** Run Held–Suarez plus an idealised NH
  mid-latitude mountain (wave-1/2 forcing) with more layers (e.g. 20–30, finer near the top),
  for about 300 days after spin-up. Compare four setups:
  - (a) no sponge;
  - (b) vorticity + divergence, eddies only (default);
  - (c) divergence only;
  - (d) including the zonal mean.

  Diagnostics:
  - the phase of the m = 1 geopotential or streamfunction wave with height. A westward tilt
    means propagation; no tilt with a node means a standing pattern, i.e. reflection;
  - wave amplitude near the lid;
  - zonal-mean u and T below the sponge. The differences between (b) and (d) show the
    Shepherd et al. spurious downward influence.
- The video scripts in `../video` can be reused for visual checks.

## Results

T31L30 reflection experiment (`experiments/reflection_dry_T31L30.png`):

- **Below the sponge, the stationary wave-2 amplitude, phase and eddy heat flux are almost
  identical in all four setups.** The wave peaks at about 11 km and decays above. Its phase is
  nearly vertical from 5 to 30 km, and the heat flux is close to 0 above 25 km. The wave is
  evanescent in the Held–Suarez stratosphere, where ū is weak or easterly at 60˚N. Very
  little wave activity reaches the lid, so there is little to reflect, and this setup cannot
  discriminate between the options.
- **The zonal mean shows the expected differences:**
  - `full − eddy`: more than +10 m/s everywhere above about 22 km. The zonal-mean sponge
    relaxes the upper-level easterlies to 0, and its influence reaches below the sponge
    (dashed line) to about 20 km in the tropics. This is the Shepherd et al. spurious
    downward influence.
  - `eddy − none`: up to +10 m/s in the tropics above 30 km. Damping the eddies removes the
    eddy forcing that maintains the upper-level tropical easterlies.
  - `div − eddy`: div-only stays close to no sponge.
- **Next step:** repeat with a stratosphere that allows vertical propagation, i.e. a winter
  polar vortex with westerlies, e.g. a Polvani & Kushner (2002)-type equilibrium temperature.

## Documentation changes

- The docstring for `RayleighSponge`.
- A short subsection in `docs/src/examples_3D.md` after the Held–Suarez example, showing the
  NamedTuple `forcing` with a sponge and explaining the eddy-only default and the references.
- A `CHANGELOG.md` line under `## Unreleased` (no PR number, since this is kept local).
- `SpeedyWeather` is already at `0.23.0-DEV`, so no further version bump is needed. Only the
  `SpeedyWeather` package is touched.

## Known limitations

- The sponge is linear and explicit (lagged step). A time scale shorter than about Δt would
  be unstable. We could warn about that in `initialize!`.
- The profile is in σ. With the default 8 equally spaced layers only the top layer
  (σ = 0.0625) is inside a σ = 0.2 sponge in a meaningful way (β ≈ 0.78), and the next one
  (σ = 0.1875) gets β ≈ 0.01. A useful sponge needs several layers near the top.
- Temperature is not damped. Planetary waves also carry temperature anomalies, which are
  left to diffusion and to the Held–Suarez relaxation.
- With `damp_zonal_mean = false`, the eddy damping still changes the eddy momentum flux
  convergence in the sponge. It just doesn't act on ū directly.

## Future work

- Optional damping of temperature eddies (T − T̄, m > 0) towards the zonal mean.
- An implicit (backward) version, applied after the full tendency is known, for very short
  time scales.
- A ClimaAtmos-style viscous sponge: a ∇² diffusion coefficient increasing aloft for all
  variables. This is an extension of `HyperDiffusion`, not a forcing.
- A pressure or log-pressure-height profile option.

## Open questions for review

1. OK to recommend vorticity + divergence, eddies only, as the default, with divergence-only
   as a switch?
2. Defaults: `sigma = 0.2` and `time_scale = time_scale_div = Day(1)` at the top? (These are
   placeholders to be tuned in the reflection test.)
3. OK to add NamedTuple support for `model.forcing` so the sponge can be combined with
   `HeldSuarez`?
4. Should damping of temperature eddies be part of this change or left as future work?
5. Implementation route: forcing (A, recommended), diffusion slot (B) or callback (C)? And
   should a callback prototype (C) be written first to run the reflection experiment?
