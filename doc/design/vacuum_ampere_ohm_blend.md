# Smooth transition between Ohm's law and vacuum Ampère's law for E

Status: design discussion, not yet implemented. Written 2026-07-05.

## 1. Motivation

PHARE is a hybrid PIC code: ions are kinetic, electrons are a massless
fluid, and quasineutrality is assumed. Under these assumptions the
displacement current is dropped from Ampère's law, which turns the
electric field into an **algebraic, diagnostic** quantity computed once
per substep from the generalized Ohm's law
(`src/core/numerics/ohm/ohm.hpp`):

```
E = -Ve x B - grad(Pe)/n + eta*J + hyper-resistive(J)
```

Several terms here scale like `1/n` (the Hall term inside `ideal_()`,
via `Ve = Vi - J/(ne)`, and the electron pressure term in `pressure_()`).
When a simulation contains a region of very low or zero density (e.g. a
vacuum buffer around a hybrid domain), these terms diverge and Ohm's law
is simply the wrong physics there: with no charge carriers, the electric
field should instead obey the vacuum Maxwell-Ampère law

```
dE/dt = c^2 curl(B)
```

We want a **single formulation** that reduces to the current Ohm's law
behaviour in dense plasma and to vacuum Ampère's law at low density, with
a smooth (not sharp-threshold) transition, because a hard switch on a
diagnostic quantity would create a discontinuous, non-physical E at the
switching surface and would be sensitive to noise in the density field
right at the threshold.

## 2. Current code, as-is (for reference)

- `src/core/numerics/ohm/ohm.hpp`: `Ohm::operator()` computes `Enew`
  from `n, Ve, Pe, B, J` purely algebraically, fresh every call. No
  persistent state for E.
- `src/core/numerics/ampere/ampere.hpp`: `Ampere::operator()` computes
  `J = curl(B)` (mu0 = 1, normalized units). This is the *static*
  Ampère's law (no displacement current) and is what feeds Ohm's law.
- `src/core/numerics/faraday/faraday.hpp`: evolves **B** via
  `dB/dt = -curl(E)`. B is the only field that is genuinely
  time-integrated with memory across substeps.
- `src/amr/solvers/solver_ppc.hpp`: orchestrates predictor1 / predictor2
  / corrector substeps, each doing `ampere()` then `ohm()` then
  `faraday()`.
- `src/amr/tagging/`: pluggable AMR tagging strategies
  (`tagger_factory.hpp`, `default_tagger_strategy.hpp`). The default
  strategy tags cells for refinement based on gradients of B. AMR levels
  in PHARE/SAMRAI already subcycle in time (finer levels take smaller
  `dt`), which will matter for section 6 below.
- Existing precedent for regularizing a `1/n` singularity: 
  `spatial_hyperresistive_()` in `ohm.hpp` already floors density with
  `min_density = 0.1` to avoid blowing up. This is a strong hint that
  the same class of problem has already been hit in this code.

## 3. Why not a sharp (or naively blended) switch

Ohm's law is *algebraic* (no time derivative of E); vacuum Ampère is
*dynamical* (E integrated in time, with memory). You cannot linearly mix
"the value E_Ohm" with "the value produced by integrating dE/dt=c^2 curl(B)"
without first deciding what kind of object E *is* — the two limits
disagree on that. A sharp switch also makes E discontinuous in n right
at the threshold, which is unphysical and numerically dangerous (noise
in n flickers the physics being solved).

The approach below instead makes E a genuinely evolved field
**everywhere**, obeying one ODE whose coefficients depend smoothly on
local density, and derives the blend by solving that ODE exactly rather
than by picking an ad hoc interpolation formula.

## 4. Proposed method: density-dependent relaxation of E toward Ohm's law

### 4.1 Governing equation

Per substep (treating `curl(B)` and `E_Ohm` as frozen over the substep,
consistent with how the existing predictor/corrector already freezes
source terms):

```
dE/dt = lambda(n) * [E_Ohm(n, Ve, Pe, B, J) - E]  +  c_eff^2 * curl(B)
```

- First term: electrons enforce Ohm's law at a density-dependent rate
  `lambda(n)`.
- Second term: the vacuum Ampère source, always present.

This is a linear ODE with (locally, per substep) constant coefficients,
so it can be **solved exactly** (exponential integrator / Rosenbrock-type
scheme), rather than approximated with an ad hoc blend:

```
g      = lambda(n) * dt
w      = exp(-g)
h(g)   = (1 - exp(-g)) / g          -> 1 as g -> 0   (compute via -expm1(-g)/g)

E_new  = w * E_old + (1 - w) * E_Ohm + dt * h(g) * c_eff^2 * curl(B)
```

Limits:
- `lambda -> infinity` (dense plasma): `w -> 0`, `h -> 0`
  => `E_new = E_Ohm` **exactly** — recovers today's code exactly, not
  approximately.
- `lambda -> 0` (vacuum): `w -> 1`, `h -> 1`
  => `E_new = E_old + dt * c_eff^2 * curl(B)` — the explicit leapfrog
  vacuum-Ampère update.

Note this is asymptotic, not step-function: for any *finite* `g`, there
is a residual vacuum-source leakage into the dense-plasma answer,
`E_new - E_Ohm ~ h(g) * dt * c_eff^2 * curl(B) ~ dt * c_eff^2 * curl(B) / g`
for large `g` (confirmed numerically: the deviation from `E_Ohm` scales
as exactly `1/g`). This means how "dense" is dense enough in practice
depends on the chosen `c_eff` too — a larger `c_eff` needs a
correspondingly larger `g` (i.e. `n0`/`p`) to keep this residual
negligible relative to `E_Ohm`'s own scale. Not a flaw, but a coupling
between parameters worth keeping in mind when picking `n0`, `p`, `c_eff`
together, and something the standalone ODE unit test checks explicitly
(section 8.3) rather than asserting exact recovery at finite `g`.

**Numerical hazard distinct from the analytic limit:** the "vacuum limit
is protected against a bad `E_Ohm`" argument (section 4.3) only holds if
`E_Ohm` is *finite*, even if huge. If `E_Ohm` ever actually evaluates to
`NaN`/`Inf` (e.g. a genuine `x/0` if density is not floored at exactly
`n=0`), then `(1-w) * E_Ohm = 0 * NaN = NaN` in IEEE floating point —
the weight does **not** save you, despite the correct exact-arithmetic
limit. This makes flooring `n` inside Ohm's law's singular terms
(section 4.3) a **hard correctness requirement**, not an optional
safety margin. Confirmed and encoded as a unit test in section 8.3's
implementation.
- In between: one smooth formula, no branching.

This scheme is **unconditionally stable in lambda**: however stiff the
dense-plasma relaxation is, there is no timestep restriction from it
(the stiffness is solved analytically inside `w` and `h`). The only
remaining stability constraint is the ordinary explicit-wave CFL,
`dt < dx / c_eff`, and it is only active where `h(g)` is not small,
i.e. in the low-density band itself (see section 6 on subcycling).

### 4.2 Choosing lambda(n) — a trap to avoid

The physically tempting choice is `lambda(n) ~ omega_pe(n) ~ sqrt(n)`
(the real electron relaxation rate). **This does not work.** `E_Ohm`
already contains terms singular like `1/n` (Hall term, pressure term).
For `(1-w)*E_Ohm` to stay bounded as `n -> 0`, since `(1-w) ~ lambda*dt`
for small `g`, we need `lambda(n)` to vanish *faster* than `E_Ohm`
diverges, i.e. `lambda(n) = o(n)`. But `sqrt(n)/n = 1/sqrt(n) ->
infinity`, so using the physical plasma frequency directly would let
the Hall/pressure singularity leak straight through the blend.

Conclusion: `lambda(n)` must be treated as a **numerical/asymptotic-
preserving control parameter**, not a literal physical rate. Its only
job is to enforce the two correct limits with a decay rate fast enough
to tame the existing singularities.

Recommended concrete form:

```
g(n)      = (n / n0)^p ,   p >= 2
lambda(n) = g(n) / dt
```

- `n0`: transition density (physical/numerical knob — set from where
  Ohm's law starts being untrustworthy, e.g. related to where the Hall
  term or electron pressure term would otherwise dominate unreasonably).
- `p >= 2`: guarantees `lambda(n)/n -> 0` as `n -> 0`, killing the `1/n`
  singularities in the blend with margin. Recommend `p = 2` as default.
- Defining `g` this way (not baking `dt` into `n0`) makes `g`, hence
  `w` and `h`, **independent of dt**. This matters directly for PHARE's
  AMR: different levels run at different local `dt` (subcycling), and
  we don't want to have to retune `n0`/`p` per level.

### 4.3 Combine with an existing safety net: floor n inside Ohm's law

Independent of the blend, also floor `n` inside the singular terms of
`ideal_()` (Hall term, via `Ve`) and `pressure_()` in `ohm.hpp`, exactly
like `spatial_hyperresistive_()` already does with `min_density = 0.1`.
This keeps `E_Ohm` itself finite (if large) rather than truly divergent,
so even a floored-but-large `E_Ohm` cannot leak into the vacuum solution
at full strength. This is belt-and-suspenders: it relaxes how much
weight `p` alone has to carry, and reuses a pattern already accepted in
this codebase.

### 4.4 The new parameter c_eff (effective/reduced speed of light)

The current hybrid normalization has **no notion of light speed** —
it is eliminated entirely by dropping the displacement current. To use
`c_eff^2 * curl(B)` we must introduce one.

Requirement expressed by the user: c_eff must be **user-settable**, as
a normalized value where the Alfvén speed v_A = 1 (i.e. the same velocity
normalization already used throughout the code). Physically
`c / v_A` is huge (10^2 - 10^4), which would force `dt < dx / c_eff`
to be prohibitively small if applied globally — hence the explicit
request to subcycle rather than shrink the global `dt` (section 6).

Proposed interface: a new dict entry alongside the existing `ohm` dict
entries (`resistivity`, `hyper_resistivity`, ...), e.g.

```
dict["ohm"]["vacuum_light_speed"]   # in units of v_A = 1
```

read the same way `OhmInfo::FROM` currently reads `resistivity`. Default
should probably be "off" (no vacuum-Ampère contribution at all, current
behaviour) unless explicitly set, to keep this feature strictly opt-in
until validated.

## 5. Architectural consequence: E needs memory

Today, `Enew` is a fresh output of `Ohm::operator()` every call, with
no dependency on the previous value of E — the code just never needed
`E_old`. The scheme in section 4 needs `E_old` explicitly (it appears
in the `w * E_old` term). This means:

- E must become a **persisted state field**, carried across predictor1
  / predictor2 / corrector substeps, the same way B already is.
- The new blended update needs access to both the previous E and the
  freshly computed `E_Ohm` (i.e. today's `Ohm::operator()` output can
  stay almost as-is and just becomes one ingredient, not the final
  answer).
- Likely a new small class, e.g. `VacuumAmpereRelax` or folded into an
  extended `Ohm`, that takes `E_old, E_Ohm, curl(B), n, dt` and produces
  `E_new` per the formula in 4.1. `curl(B)` is already computed by the
  existing `Ampere` class, so it can be reused directly rather than
  recomputed.
- Needs boundary conditions / ghost handling for E to be defined
  consistently as a real evolving field, which previously only mattered
  insofar as Ohm's law is local and algebraic.

## 6. Subcycling only the low-density cells

The ETD scheme in 4.1 removes stiffness from `lambda(n)` but **not**
from the explicit wave term `dt * h(g) * c_eff^2 * curl(B)`. Stability
of that term requires `dt < dx / c_eff` — but only where `h(g)` is not
negligible, i.e. in the low-density band itself (`h(g) -> 0` quickly as
`n` grows past `n0`). This is spatially localized by construction,
which is exactly why subcycling only those cells (rather than shrinking
the global `dt`) is attractive.

Preferred approach: **reuse PHARE's existing AMR tagging + subcycling
machinery** instead of building a bespoke local-time-stepping engine.

- PHARE already supports pluggable tagging strategies
  (`src/amr/tagging/tagger_factory.hpp`,
  `default_tagger_strategy.hpp`), and AMR levels in
  SAMRAI already subcycle in time (finer levels take a fraction of the
  coarse `dt`, governed by the refinement ratio).
- Add a new tagging criterion (or extend the existing strategy) that
  tags cells for refinement where `n` is below some multiple of `n0`
  (i.e. where `h(g)` — or equivalently `g(n)` — is not yet negligible).
  This ties the *spatial* refinement criterion directly to the same
  `n0` used in the E-blend, so the two are self-consistent by
  construction rather than two independently-tuned thresholds.
- The finer level automatically gets the smaller `dt` it needs to
  satisfy `dt < dx_fine / c_eff` via the existing subcycling
  infrastructure — no new time-stepping code path required, "just" a
  new density-based tagger.
- Caveats to verify once this is prototyped:
  - Does the refinement ratio give a small enough `dx_fine` and
    small enough subcycled `dt_fine` to satisfy the c_eff CFL for the
    values of `c_eff` we actually want to use? If not, may need several
    refinement levels stacked in the vacuum region, or a dedicated
    (non-spatial) local subcycle specifically for the E/B pair within
    the tagged cells regardless of AMR refinement.
  - Regridding cadence: density-based tagging means the refined region
    moves/grows/shrinks as the plasma evolves (e.g. an expanding
    density front), unlike today's more static geometry-driven
    refinement in typical setups. Need to check regrid frequency is
    sufficient to track this without lag.
  - Interaction between the coarse/fine boundary and a physically
    meaningful E/B field — today's inter-level coupling was designed
    around B (and algebraic E); need to confirm the same
    prolongation/restriction operators are adequate for a now-dynamical
    E with real wave content, or whether reflections appear at the
    coarse-fine interface (a known issue in AMR wave problems, "AMR
    interface reflection").

## 7. Numerical safety notes

- Use `-expm1(-g)/g` (or a Taylor series `1 - g/2 + g^2/6 - ...` for
  very small `g`) to evaluate `h(g)`, never the naive
  `(1 - exp(-g))/g`, to avoid catastrophic cancellation as `g -> 0`.
- `w = exp(-g)` is well-behaved for all `g >= 0` (bounded in [0,1]), no
  special-casing needed there.
- Floor `n` inside `ideal_()`/`pressure_()` per section 4.3, independent
  of the blend weight, so `E_Ohm` itself is never literally `inf`/`nan`
  even transiently.
- Verify `(1-w) * E_Ohm` stays bounded numerically (not just
  analytically) as `n -> 0` in a standalone scalar test before wiring
  into the full solver (see section 8).

## 8. Testing plan

1. **Pure vacuum** (`n = 0` everywhere): the scheme should reduce
   exactly to source-free Maxwell. Verify against the analytic vacuum
   dispersion relation `omega = c_eff * k` for a plane wave, and check
   energy conservation (no artificial damping/growth) over many
   periods.
2. **Uniform dense plasma regression**: with `n >> n0` everywhere (large
   `g`, `w ~ 0`), output must match the current Ohm's-law-only code to
   round-off. Reuse/extend any existing whistler dispersion test as a
   regression baseline.
3. **Scalar ODE unit test** (no grid, no solver): integrate the 4.1 ODE
   for a few representative `(n, dt, c_eff)` triples against a
   brute-force fine-substepped explicit integration of the same ODE,
   to validate the exact-solution formula and its numerical evaluation
   (`w`, `h(g)`) independent of the rest of the code.
4. **Density interface / ramp** (dense plasma next to vacuum, e.g. a
   smooth or sharp density profile): check for spurious reflections or
   heating at the transition layer, boundedness of E and B, and
   consistency of `div(E)` with charge (Gauss's law) if E is now really
   being evolved — this was never a concern before since E was never a
   propagated field in this code.
5. **Sensitivity study**: vary `n0` and `p` and confirm results converge
   and are not overly sensitive once solidly inside either asymptotic
   regime.
6. **Stability/CFL sweep**: sweep density down toward zero (including
   exactly zero) and confirm no blow-up/NaNs — this directly targets
   the motivating bug (Ohm's law dividing by `n ~ 0`).
7. **Subcycling / tagging test**: confirm the density-based tagger
   correctly refines the low-density band, that the resulting `dt_fine`
   satisfies the c_eff CFL there, and that a moving density front is
   tracked by regridding without lag or interface artifacts.
8. **End-to-end physical case**: rerun an existing hybrid test problem
   with an added low-density vacuum buffer around it, confirm plasma-
   region physics is unchanged from the no-buffer run, and confirm the
   buffer no longer produces the current divide-by-`n` failure.

## 9. Open parameters / decisions to make before implementing

- `n0`: transition density (physical choice, problem-dependent or a
  code default).
- `p`: falloff power in `g(n) = (n/n0)^p`. Recommend `p = 2`.
- Density floor value used inside `ideal_()`/`pressure_()` (reuse
  `min_density` pattern from `spatial_hyperresistive_()`?).
- `c_eff`: user-settable via `dict["ohm"]["vacuum_light_speed"]` (name
  tbd), normalized to v_A = 1. Feature should default to off (i.e. this
  whole mechanism opt-in) until validated.
- Whether tagging threshold and `n0` should be literally the same
  dict value or two related-but-separate knobs.
- Whether AMR refinement alone gives sufficient `dt` reduction for
  realistic `c_eff`, or whether a dedicated local subcycle is still
  needed on top of it (section 6 caveats).

## 10. Implementation touch points (once design is settled)

- `src/core/numerics/ohm/ohm.hpp`: add density floor to `ideal_()` /
  `pressure_()`; `Ohm` output becomes an intermediate `E_Ohm`, not the
  final `E`.
- New small component (name tbd, e.g. `vacuum_ampere_relax.hpp`) that
  implements the 4.1 closed-form update given `E_old`, `E_Ohm`,
  `curl(B)` (from the existing `Ampere`), `n`, `dt`, `n0`, `p`, `c_eff`.
- `src/amr/solvers/solver_ppc.hpp`: orchestration changes so E has
  persistent state across predictor1/predictor2/corrector and the new
  relax step runs after `ohm()` and before `faraday()`.
- `src/amr/tagging/`: new density-based tagging criterion (new
  strategy or extend `default_tagger_strategy.hpp`), wired through
  `tagger_factory.hpp`.
- `initializer` dict plumbing for the new parameters (pattern to
  follow: `OhmInfo::FROM` in `ohm.hpp`).

## 11. Next steps

Discuss and pin down section 9's open parameters, then decide on a
prototyping order — most likely: (a) the standalone scalar ODE test
(item 8.3) to validate the exact-solution formula in isolation, (b) a
1D toy density-ramp problem exercising the full blend without AMR
subcycling, then (c) wiring into the real solver plus the AMR tagging
piece for subcycling.

### Progress

- **(a) done.** `src/core/numerics/ohm/vacuum_ampere_relax.hpp` implements
  the scalar closed form from section 4.1 (`VacuumAmpereRelax`), and
  `tests/core/numerics/vacuum_ampere_relax/test_main.cpp` validates it:
  exact recovery of the vacuum update at `n=0` even with a huge-but-
  finite poisoned `E_Ohm`; explicit confirmation that a `NaN`/`Inf`
  `E_Ohm` still poisons the result at `n=0` (the numerical hazard from
  the section 4.1 addendum); the `1/g` scaling of the dense-limit
  residual; agreement with a brute-force fine-stepped Euler integration
  of the same ODE; and a direct comparison of the recommended
  `g(n)=(n/n0)^2` falloff against the physically-tempting-but-wrong
  `lambda ~ sqrt(n)`, showing the latter fails to suppress the `1/n`
  Ohm's-law singularity as `n -> 0` while the former does (section 4.2's
  pitfall). Registered in the build via
  `res/cmake/test.cmake`.
- Not yet done: the density floor inside `ideal_()`/`pressure_()` in
  `ohm.hpp` (section 4.3) — still needed before this component is wired
  into anything real, since the unit tests above show the blend alone
  does not protect against a genuinely non-finite `E_Ohm`.

- **(b) done.** `doc/design/prototypes/vacuum_ampere_blend_1d.py` +
  `run_scenarios.py`: a 1D (Yee-staggered, periodic) toy with a
  Hall-term-only Ohm's law (`Ve = -J/(n_floored e)`, immobile ions —
  chosen specifically to reproduce the real `1/n` Hall singularity, not
  a watered-down model) and the `VacuumAmpereRelax` blend. Three
  scenarios all pass: uniform dense plasma matches plain Ohm's law to
  1.4e-5 relative; uniform vacuum wave speed matches `c_eff*k` to 0.26%;
  a wave packet crossing a dense-vacuum-dense trench (n: 1 -> 1e-3)
  stays bounded for the full ~670k-step crossing, with a plausible
  ~57/43 transmitted/reflected energy split. Full report with plots:
  see the artifact linked from this session, or regenerate via
  `run_scenarios.py` + the plotting snippet in the session transcript.

  Getting scenario C to pass surfaced three numerical findings that are
  **about the toy's own time integration, not about the blend formula**,
  but are worth keeping since they'll matter again in step (c):

  1. A naive single-substep sequential update (E fully from current B,
     then B fully from that new E — regardless of which is computed
     first) is *unconditionally, weakly unstable* for the bare
     Hall/whistler rotation, at every wavenumber. Confirmed both
     analytically (its single-mode transfer matrix has eigenvalues
     `1 +- i*Omega`, magnitude `sqrt(1+Omega^2) > 1` for any nonzero
     `Omega`) and empirically (a real, if slow-onset, exponential
     blow-up — a high-accuracy exact-time-integration reference
     confirmed the *spatial* discretization itself is fine, isolating
     the bug to time-stepping). This is presumably why real PHARE's
     predictor-corrector substep structure is load-bearing for this
     term and not just an accuracy nicety — a single bare substep
     seems to not be enough.
  2. The fix is standard single-level Yee time-staggering (advance B
     using the current E, then E using the just-updated B) — verified
     to have exactly as many eigenvalues as physical degrees of freedom
     (no spurious extra mode, unlike a 2-time-level leapfrog-for-B
     variant that was tried and discarded), magnitude-1 stability for
     the algebraic/dense limit, and vacuum dispersion matching analytic
     to ~0.01% in a single-mode check. The residual (very weak)
     Hall-term growth this leaves is fully controlled by a *small*
     hyper-resistive term matching `ohm.hpp`'s existing
     `constant_hyperresistive_` (`nu=1e-4` was enough here) — meaning
     that term is apparently numerically load-bearing for explicit
     whistler integration, not only a physics nicety.
  3. Independent of both of the above: a too-sharp density gradient is
     *itself* a source of numerical instability (a trench ramp resolved
     by only ~20 cells reliably blew up regardless of the fixes above,
     even at a much shallower density contrast; widening the ramp to
     ~100 cells fixed it completely at the *original* deep contrast).
     This directly reinforces section 6's plan to spatially refine the
     transition band — not only for the `c_eff` CFL, but to adequately
     resolve the density gradient itself.

  None of this changes the blend formula (section 4.1) or its
  parameters — it's entirely about what the surrounding field solver
  needs to look like for the blend to be exercised fairly on a grid.
  `vacuum_ampere_relax.hpp` and its unit tests are unaffected.

- **(c), first slice done: the section 4.3 density floor.** Landed as a
  new, opt-in (`default = 0.0`, i.e. no behavior change unless set)
  `min_density` dict parameter in two places, since the singular `1/n`
  terms turned out to live in two different files:
  - `ohm.hpp`'s `OhmInfo`/`pressure_()`: floors `n` before dividing in
    the electron pressure term (`dict["ohm"]["min_density"]` from
    Python, i.e. `Simulation(..., min_density=...)`).
  - `electrons.hpp`'s `StandardHybridElectronFluxComputer`: floors `Ne`
    before dividing in `computeBulkVelocity` — this is the *actual* Hall
    term singularity (`Ve = Vi - J/(ne)`), which does not live in
    `ohm.hpp` at all (`ohm.hpp`'s `ideal_()` takes `Ve` as an
    already-computed input). `dict["electrons"]["min_density"]` from
    Python, i.e. `ElectronModel(..., min_density=...)`.

  Both wired end-to-end through `pyphare` (`simulation.py`,
  `electron_model.py`, `initialize/hybrid.py`) with validation
  (non-negative) matching the existing `resistivity`/`Te` pattern.
  Verified: `test-ohm` (6/6) and `test-electrons` (45/45) still pass
  unmodified with the new parameters at their default (no-op) value —
  built and run locally against a SAMRAI install already cached on this
  machine, not just compiled in isolation.

  Deliberately *not* done here: unifying this with the pre-existing,
  unrelated hardcoded `min_density = 0.1` inside `ohm.hpp`'s
  `spatial_hyperresistive_()` — that's a different, already-working
  mechanism, out of scope for this change.

- Next: the rest of (c) — the actual `VacuumAmpereRelax` wiring into
  `solver_ppc.hpp` (currently `Ohm`'s output is used as final `E`
  directly; needs to become an intermediate `E_Ohm` feeding the relax
  step, with `E` gaining persistent state across predictor1/predictor2/
  corrector), plus the AMR density-based tagging piece for subcycling.
  Still need to decide whether PHARE's existing predictor-corrector
  substep structure already provides the time-staggering step (b)'s
  finding (1) needs, or whether the relax step's placement needs
  adjusting.
