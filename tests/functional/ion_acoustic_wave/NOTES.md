# iaw_polytropic.py debugging notes (2026-07-03, session paused mid-investigation)

Context: new polytropic electron pressure closure in `src/core/data/electrons/electrons.hpp`
(`PolytropicElectronPressureClosure`), exercised by this test. Originally reported as
"compiles, segfaults at the first of execution".

**⚠️ CURRENT REPO STATE IS NOT CLEAN — read section 6 before doing anything else.**
Heavy `std::cerr` debug instrumentation is left in several files to support the bisection
below. `git diff --stat` shows more files touched than in the previous session.

## 1. Driver-script bugs (fixed, by the user, in `iaw_polytropic.py`)

- `startMPI()` imported commented-out and never called -> abort before `MPI_Init`. Fixed.
- `config()` never returned the `Simulation` -> `Simulator(config())` got `None`. Fixed.

## 2. Fix #1 (done, solid): electron pressure `Pe` ghosts were never filled

Wired `Pe`/`Te` into the AMR hybrid messenger like `J` (see memory file
`hybrid-messenger-ghost-fill-architecture`). Added non-throwing `pressureResource()` accessor
chain. **Kept, solid.**

## 3. Fix #2 (done, solid): electron bulk velocity `Ve` ghosts were also never filled

Split `Electrons::update()` into `updateMoments()`/`updatePressure()`; `SolverPPC::
update_electrons()` now: loop `updateMoments` -> `fillElectronVelocityGhosts` -> loop
`updatePressure` -> `fillElectronPressureGhosts`. **Kept, solid, verified `Vex` numerically
sane (non-NaN) through every substep of every timestep tested so far.**

## 4. Fix #3 (done, solid, restored 2026-07-03): T_Eq_ sign

`∂T/∂t = -V·∇T - (γ-1)T∇·V`. `T_Eq_` now correctly does
`Te - dt*(advection_(...) + compression_(...))`. **This session restored the real formula**
(previous session had temporarily zeroed both terms for bisection — that's undone now).
**Empirically ruled out as the crash cause**: re-tested with both terms multiplied by 0
(temperature evolution fully disabled) — the exact same crash still occurred. So the FTCS/
forward-Euler-instability theory from the previous session is **dead**; do not pursue it
further.

## 5. Fix #4 (done, solid): gamma/Gamma kwarg-case mismatch

`iaw_polytropic.py` was calling `ElectronModel(closure="polytropic", Pe=Pe, gamma=gamma_e)`
(lowercase `gamma`) but `PolytropicClosure.__init__` (`pyphare/pharein/electron_model.py`) only
reads `kwargs.get("Gamma", ...)` (capital G) — so the user's `gamma_e=3` was silently ignored
and the closure always ran with the default `Gamma=1.66`, while the test's analytic sound speed
(`cs = sqrt(gamma_e*Te + gamma_i*Ti)`) assumed `gamma_e=3`. **Fixed the call site** to
`Gamma=gamma_e`. This uncovered a **second, previously-masked bug**: with `Gamma` now an `int`
(3) instead of the old default `float` (1.66), `pyphare/pharein/initialize/hybrid.py`'s
`populateDict` loop only handled `str`/`float`/`Callable` dict_path entries, raising
`ValueError: acceptable entries should be int, float or collable` for the `int`. **Fixed** by
widening the check to `isinstance(item[1], (float, int))`. Both fixes are real, solid, unrelated
to the crash below, and should be kept.

(Verified `Gamma` really does thread through to the C++ closure's `gamma_` member correctly via
`PHAREDict` — `electrons.hpp`'s in-class `gamma_ = 5./3.` is just a fallback default always
overridden by the constructor initializer list `gamma_{dict["pressure_closure"]["Gamma"]...}`.)

## 6. ⚠️ WHERE THIS SESSION WAS PAUSED — the real, narrowed-down bug

After all fixes above, the exact same `AMRToLocal: Assertion 'local >= 0' failed` crash (particle
position -> `INT_MIN`) still occurs on the very first `advance()`. **This has now been root-caused
much more precisely than before**, via a long instrumentation bisection (`std::cerr` prints
added at every stage of the E-field computation pipeline — see section 7 for exactly what's
currently instrumented and needs removing):

**The bug: `fillElectronPressureGhosts` (the AMR ghost-fill call for `Pe`) corrupts the ENTIRE
`Pe` array — domain AND ghosts — to NaN, the very first time it's called inside a real
`predictor1_`.** Confirmed via prints immediately before/after:
- Right after `computePressure()` returns (writes `Pe` from `Te`/`N` domain values):
  `Pe[0.0996233,0.100377]` — sane.
- Immediately after `fromCoarser.fillElectronPressureGhosts(...)` returns (same substep, no other
  code in between): `Pe[nan,nan]` — **the entire array**, not just ghost cells.

Traced deeper into `FieldData::copy(source, overlap)` (`src/amr/data/field/field_data.hpp`):
added a print keyed on `field.name()=="Pe"`. Result: `this == &fieldSource` (literally the same
C++ `FieldData` object — expected, since Pe's ghost refiner is a **self-fill**, dst==src=="Pe",
same as how J/E/B are wired) — but **both `dst` and `src` already read all-NaN at entry to
`copy()`**, i.e. the corruption happens *before* `copy()` even runs, inside SAMRAI's schedule
machinery itself (schedule setup / internal scratch allocation), not in our copy/refine code.
Also confirmed the custom `FieldRefineOperator::refine()` (`field_refine_operator.hpp`) is
**never called at all** for this test (added a print there — it never fires) — ruling out our
own interpolation policy code as the cause; this is a pure same-level periodic-ghost transfer
that should be a plain copy, and something in setting that copy up wipes the buffer.

**The likely root cause, found by comparing against every other scalar-field ghost-fill in the
whole codebase (`grep`-swept `hybrid_messenger_strategy.hpp` and `mhd_messenger.hpp`):**
`Pe`'s wiring uses `GhostRefinerPool::addStaticRefiners` (self-fill, dst==src, **no time
interpolation**) with the plain scalar `fieldRefineOp_` (`FieldRefineOperator`, not a
Vec/TensorField wrapper). **This exact combination — bare scalar `Field` + `GhostRefinerPool`
(`RefinerType::GhostField`, the coarser-level-aware 4-arg `createSchedule`) + `addStaticRefiners`
self-fill — has *never* been used anywhere else in PHARE before this branch.** Every other
scalar quantity that goes through a `GhostRefinerPool` (`Ni`/`rho` in hybrid, `rho`/`Etot` in
MHD) does so via `addTimeRefiners` (with an "Old" shadow field, e.g. `NiOld_`, kept in sync via a
plain `NiOld_.copyData(Ni)` call once per full step inside `firstStep()`) — **never**
`addStaticRefiners`. The only existing `addStaticRefiners` self-fill users of `GhostRefinerPool`
are all VecFields (E, B, J, MHD momentum/magFluxes), which go through
`TensorFieldRefineOperator`/`TensorFieldData`, a structurally different C++ class from the plain
scalar `FieldData` that `Pe` uses. `Ve` (this branch's *other* new ghost-filled quantity) dodges
this bug entirely because it's a `VecField` (goes through the proven Tensor path) — this is why
`Ve` has worked flawlessly throughout while `Pe` (the only bare-scalar `addStaticRefiners`
self-fill in the entire codebase) is broken. So this is **not** a bug in the fix's design, it's
a **previously-latent, never-before-exercised gap in PHARE's SAMRAI ghost-fill integration for
scalar fields specifically wired the "static self-fill" way** — most likely inside
`SAMRAI::xfer::RefineSchedule`'s internal scratch-patch-data handling when
`registerRefine(id, id, id, refineOp, fillPattern)` is called with a **scalar** `FieldDataFactory`
and all three ids identical (works fine for the tensor/vecfield factory, apparently does not for
the scalar one) — but this wasn't confirmed at the SAMRAI-internals level (would need actually
stepping through `RefineSchedule::fillData()`, not just our own code, to nail down exactly which
allocate/copy step wipes the buffer).

**Recommended next step (not yet implemented, needs a decision — see below)**: route `Pe`
through `GhostRefinerPool::addTimeRefiners` instead of `addStaticRefiners`, mirroring `Ni`/`rho`
exactly: add a `PeOld_` shadow field (`FieldT PeOld_{stratName + "_PeOld",
core::HybridQuantity::Scalar::P}` alongside `NiOld_`), register/allocate it the same way, sync it
once per full step in `firstStep()` (`PeOld_.copyData(Pe)`, mirroring the existing
`NiOld_.copyData(Ni)` call), and change the `electronPressureGhostsRefiners_.addStaticRefiners(...)`
call (hybrid_hybrid_messenger_strategy.hpp ~line 839) to `addTimeRefiners(info->
ghostElectronPressure, info->modelElectronPressure, PeOld_.name(), fieldRefineOp_, fieldTimeOp_,
...)`. For a **single-level** simulation (this test), same-level ghost fills should use the
"new"/current data directly regardless of the Old field's staleness (time-interpolation only
matters for genuine cross-level fills against a coarser level at a different time) — so this
should be safe and correct for this test. **Caveat for future multi-level AMR use of this
closure**: `Pe` is recomputed 3x per full step (predictor1_/predictor2_/corrector_) but
`firstStep()` only syncs `PeOld_` once per full step, so a genuinely refined finer level doing
cross-level time-interpolation against `Pe`/`PeOld_` mid-step would interpolate against a
stale bracket — the same tension already noted (and deliberately avoided, by using a different
mechanism) for `Ve` in fix #2 above. This is a real but currently out-of-scope wrinkle since the
failing test is single-level.

**This was flagged to the user rather than implemented outright**, since adding a new persistent
field + wiring it into the `firstStep()`/sync lifecycle is a real (if small) architectural
change, not a pure bug fix, and there may be a cleaner alternative (e.g. fixing the actual SAMRAI
scalar-self-fill gap, or finding another already-working scalar ghost-fill pattern to mirror
instead).

## 7. Debug instrumentation currently left in the tree (must be removed once the fix lands)

- `src/core/data/electrons/electrons.hpp`: `#include <iostream>` + several `std::cerr` prints in
  `computePressure` (pre/post `T_Eq_`, array sizes).
- `src/amr/solvers/solver_ppc.hpp`: `#include <algorithm>`/`#include <iostream>` + a
  `std::cerr` print in `average_()` (Ex/Bx min/max) + a print loop after
  `fillElectronPressureGhosts` in `update_electrons()`.
- `src/amr/solvers/solver_hybrid_field_evolvers.hpp`: `#include <algorithm>`/`#include <iostream>`
  + prints of ohm's inputs (n/Pe/Ve/B/J) and resulting `Ex` in `OhmLevelTransformer::operator()`.
- `src/amr/data/field/refine/field_refine_operator.hpp`: `#include <iostream>` + a print in
  `FieldRefineOperator::refine()` (confirmed never fires for this test — safe to remove, but keep
  the finding in mind).
- `src/amr/data/field/field_data.hpp`: `#include <algorithm>`/`#include <iostream>` + prints in
  both `copy()` overloads keyed on `field.name() == "Pe"`.

**To resume**: re-run
`PYTHONPATH=/home/smets/far/builds/debug/close:/home/smets/far/PHARE/pyphare python3 -u iaw_polytropic.py`
after `make cpp_1_1_2` in `/home/smets/far/builds/debug/close`. Once a fix for section 6 is
chosen and implemented, remove all the instrumentation in section 7 and confirm the test runs to
completion cleanly.
