# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PHARE is a Parallel Hybrid Particle-In-Cell (PIC) code with Adaptive Mesh Refinement (AMR). It solves
the Vlasov equation for an arbitrary number of ion populations kinetically (particles), while electrons
are a single fluid whose momentum equation gives the electric field under quasineutrality (Ohm's law
closure). An MHD solver also exists alongside the hybrid (PIC) solver. AMR is provided by
[SAMRAI](https://github.com/llnl/samrai). Core language is C++20; simulations are configured and driven
from Python (pybind11 bindings).

## Build

```bash
mkdir build && cd build
cmake .. -DSAMRAI_ROOT=/path/to/SAMRAI/install   # omit -DSAMRAI_ROOT to build SAMRAI from source
make -j$(nproc)
```

Useful CMake options (see `res/cmake/options.cmake` for the full list):
- `-DdevMode=ON` — strict warnings (`-Wall -Wextra -Werror` etc.), enables ccache if found.
- `-Dtest=ON` (default ON) — build the gtest/ctest suite; `-DtestMPI=ON` runs tests under `mpirun`.
- `-Dasan=ON` / `-Dubsan=ON` — sanitizer builds.
- `-Dbench=ON` — build Google Benchmark targets.
- `-Dphare_configurator=ON` — auto-detect a working compile setup via `tools/config`.
- `-DPHARE_MPI_PROCS=N` — number of ranks used by `add_phare_test`/MPI tests (default 1, or 2 with `testMPI`).

Python dependencies: `python3 -m pip install -r requirements.txt`. The Python package `pyphare` is used
directly from the source tree (no install step needed) — CMake sets `PYTHONPATH` for tests to
`<build_dir>:<repo_root>/pyphare`.

## Tests

Tests are registered with CTest through custom wrapper functions defined in `res/cmake/def.cmake`
(`add_phare_test`, `add_no_mpi_phare_test`, `add_python3_test`, `add_no_mpi_python3_test`,
`add_mpi_python3_test`) and enabled per-directory in `res/cmake/test.cmake`. Every test subdirectory
under `tests/` has its own `CMakeLists.txt`.

```bash
cd build
ctest --output-on-failure                 # full suite
ctest -R test-ndarray --output-on-failure # single test by CTest name
ctest -j4 --output-on-failure             # parallel
./tests/core/data/ndarray/test-ndarray    # run a single gtest binary directly (supports gtest filters, e.g. --gtest_filter=...)
```

Python-only tests are prefixed `py3_` in ctest (e.g. `py3_test-pyphare-box`), and can also be run
directly with `python3 <file>.py` from the relevant `tests/` or `pyphare/pyphare_tests/` directory once
`PYTHONPATH` includes the build dir and `pyphare/`.

Test layout mirrors `src/`:
- `tests/core`, `tests/amr`, `tests/diagnostic`, `tests/initializer`, `tests/simulator` — unit/integration
  tests for the matching `src/` subsystem, C++ (gtest) and/or Python.
- `tests/functional/*` — full end-to-end simulations (Harris sheet, Alfvén wave, shocks, dispersion, MHD
  variants, etc.), each a Python script using `pyphare` to configure and run a `Simulator`, gated behind
  `PHARE_EXEC_LEVEL_MIN/MAX` and often requiring HighFive/MPI (see each `CMakeLists.txt` for exec level
  and rank count).

CI (`.github/workflows/cmake_ubuntu.yml`) builds with `-DdevMode=ON -Dbench=ON -Dphare_configurator=ON`
and runs `ctest -j2 --output-on-failure` — use this as the reference invocation for a full local check.

## Architecture

PHARE is organized as compile-time-configured layers, glued together by a single template parameter
pack (`SimOpts`: `dimension`, `interp_order`, `nbRefinedPart`). Each layer exposes a `PHARE_Types<opts>`
struct that resolves concrete types for that configuration; `src/simulator/phare_types.hpp` composes
`core::PHARE_Types`, `amr::PHARE_Types`, and `solver::PHARE_Types` into the final `PHARE::PHARE_Types<opts>`
used by `Simulator<opts>` (`src/simulator/simulator.hpp`). This is why many headers are templated on
`opts`/dimension/interp_order rather than using runtime polymorphism for hot-path types.

- **`src/core`** — physics- and AMR-agnostic building blocks: grids/fields/particles (`core/data`),
  numerical kernels (`core/numerics`: pushers, interpolators, Ampère/Faraday/Ohm solvers), and the two
  physical state representations (`core/models/hybrid_state.hpp`, `core/models/mhd_state.hpp`) built on
  `core/hybrid` and `core/mhd` quantity definitions. No SAMRAI dependency here.
- **`src/initializer`** — bridges user-facing configuration into C++. `PHAREDict` (`data_provider.hpp`)
  is a dynamically-typed dictionary; `python_data_provider.hpp` fills it from the Python simulation
  description, `restart_data_provider.hpp` from a restart file.
- **`src/amr`** — SAMRAI integration: patch data/variables for fields and particles (`amr/data`),
  inter-patch/inter-level communication (`amr/messengers`), resource allocation on patches
  (`amr/resources_manager`), level creation (`amr/level_initializer`), refinement criteria
  (`amr/tagging`), load balancing (`amr/load_balancing`), and `amr/solvers` (hybrid PPC solver, MHD
  solver) coordinated by `amr/multiphysics_integrator.hpp` (advances/synchronizes multiple physical
  models across the patch hierarchy).
- **`src/simulator`** — `Simulator<opts>` ties core+amr+solver types together and implements
  `initialize()`/`advance(dt)`; `ISimulator` is the non-templated interface exposed to Python.
- **`src/diagnostic`**, **`src/hdf5`**, **`src/restarts`** — output: diagnostics dumps, HDF5 writers
  (HighFive), and restart read/write.
- **`src/python3`**, **`src/phare`** — pybind11 module and the standalone `phare` executable.
- **`pyphare/pyphare`** — the Python side: `pharein` (declarative simulation setup — `Simulation`,
  particle/fluid models, diagnostics, load balancer config — this is what user input scripts import as
  `ph`), `simulator` (`Simulator`/`startMPI`, drives the compiled `ISimulator` via `pyphare.cpp`
  bindings), `pharesee` (post-processing/analysis of run output, e.g. `Run`), `core`/`data` (shared
  numerics/data helpers used by both simulation setup and analysis).

A typical simulation input (see `tests/functional/harris/harris_2d.py`) is a Python script that builds a
`ph.Simulation(...)`, defines initial-condition callables (density, bulk velocity, B-field, temperature),
and hands them to `pyphare.simulator.simulator.Simulator`, which calls into the C++ `Simulator<opts>`
through pybind11.

## Python conventions (from `doc/conventions.md`)

Do not import `h5py`, `mpi4py`, or `scipy.optimize` at module (top) scope in Python files under
`pyphare`/`tests`. These pull in system libraries (libhdf5/libmpi) that may mismatch what PHARE was
built against, and can break simply importing/scanning a file before any simulation runs. Import them
inside the function that needs them instead. `numpy` is fine at top scope.
