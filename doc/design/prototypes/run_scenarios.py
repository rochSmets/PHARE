"""
Driver for the three validation scenarios described in
doc/design/vacuum_ampere_ohm_blend.md section 11 ("Progress", step (b)):

  A. uniform dense plasma:  blend must match plain Ohm's law
  B. uniform vacuum:        wave speed must match c_eff*k
  C. dense-vacuum-dense trench: fields must stay bounded crossing it

Run from this directory with `python3 run_scenarios.py`. Scenario C alone
takes ~2 minutes (it integrates ~670k explicit steps to let a wave packet
actually cross the trench). Results are written to scenario_results.npz
next to this script; see the design doc for the numbers this produced.
"""

import os
import time
import numpy as np
from vacuum_ampere_blend_1d import (Sim, uniform_density, trench_density, pick_dt)

Bx0 = 1.0
n_bg = 1.0
n0 = 0.05
p = 2.0
c_eff = 20.0
n_floor_eps = 1e-6

Nx = 800
L = 40.0


def gaussian_packet(x, x0, width, k):
    return np.exp(-((x - x0) / width) ** 2) * np.sin(k * (x - x0))


results = {}

# ---------------------------------------------------------------- scenario A
print("=== Scenario A: uniform dense domain, blend vs pure-Ohm ===")
n_of_x = lambda x: uniform_density(x, n_bg)
dt = pick_dt(L / Nx, Bx0, n_bg, c_eff, n0, p, n_floor_eps)
print(f"dt = {dt:.3e}")

k_mode = 2 * np.pi * 6 / L  # 6 wavelengths across the box

simBlend = Sim(Nx, L, Bx0, n_of_x, n0, p, c_eff, n_floor_eps, dt, use_blend=True)
simPure = Sim(Nx, L, Bx0, n_of_x, n0, p, c_eff, n_floor_eps, dt, use_blend=False)
for s in (simBlend, simPure):
    s.By[:] = 1e-3 * np.sin(k_mode * s.xB)
    s.Bz[:] = 0.0

nsteps = 20000
for _ in range(nsteps):
    simBlend.step()
    simPure.step()

g_val = (n_bg / n0) ** p
print(f"g in dense region = {g_val:.3e}  (w = exp(-g) = {np.exp(-g_val):.3e})")
maxdiff = max(np.max(np.abs(simBlend.By - simPure.By)), np.max(np.abs(simBlend.Bz - simPure.Bz)),
             np.max(np.abs(simBlend.Ey - simPure.Ey)), np.max(np.abs(simBlend.Ez - simPure.Ez)))
scale = max(np.max(np.abs(simPure.By)), np.max(np.abs(simPure.Bz)))
print(f"max|blend - pureOhm| (B only) = {maxdiff:.3e}   (field scale = {scale:.3e}, relative = {maxdiff/scale:.3e})")
assert np.isfinite(simBlend.max_abs_field()) and np.isfinite(simPure.max_abs_field())
assert maxdiff / scale < 1e-3, "blend should be nearly indistinguishable from pure Ohm in the dense limit"
print("PASS\n")
results['A'] = dict(maxdiff=maxdiff, scale=scale, g_val=g_val)


# ---------------------------------------------------------------- scenario B
print("=== Scenario B: uniform vacuum domain, wave speed vs c_eff ===")
n_of_x = lambda x: uniform_density(x, 0.0)
dx_local = L / Nx
k_max = np.pi / dx_local
dt = 0.4 * 2.0 / (c_eff * k_max)

k_mode = 2 * np.pi * 6 / L
sim = Sim(Nx, L, Bx0, n_of_x, n0, p, c_eff, n_floor_eps, dt, use_blend=True)
sim.By[:] = 1e-3 * np.sin(k_mode * sim.xB)
sim.Bz[:] = 0.0

nsteps = 2000
snapshots = []
times = []
t = 0.0
for it in range(nsteps):
    sim.step()
    t += dt
    if it % 5 == 0:
        snapshots.append(sim.By.copy())
        times.append(t)
snapshots = np.array(snapshots)
times = np.array(times)

Bk = np.fft.rfft(snapshots, axis=1)
kfft = np.fft.rfftfreq(Nx, d=L / Nx) * 2 * np.pi
idx = np.argmin(np.abs(kfft - k_mode))
phase = np.unwrap(np.angle(Bk[:, idx]))
A = np.vstack([times, np.ones_like(times)]).T
slope, intercept = np.linalg.lstsq(A, phase, rcond=None)[0]
omega_measured = abs(slope)
omega_analytic = c_eff * k_mode
print(f"k = {k_mode:.4f}, omega_measured = {omega_measured:.4f}, omega_analytic (c_eff*k) = {omega_analytic:.4f}")
relerr = abs(omega_measured - omega_analytic) / omega_analytic
print(f"relative error = {relerr:.3e}")
assert np.isfinite(sim.max_abs_field())
assert relerr < 0.02, "vacuum-limit wave speed should match c_eff*k within 2%"
print("PASS\n")
results['B'] = dict(k=k_mode, omega_measured=omega_measured, omega_analytic=omega_analytic,
                    relerr=relerr, snapshots=snapshots, times=times)


# ---------------------------------------------------------------- scenario C
print("=== Scenario C: dense-vacuum-dense trench, boundedness + transmission ===")
half_width = 3.0
ramp = 5.0   # must adequately resolve the density gradient -- see design doc addendum:
             # a too-sharp ramp (e.g. ramp=1.0, ~20 cells) triggers a genuine
             # parametric numerical instability in the explicit Hall-term
             # integration, independent of the E-blend itself.
n_floor = 1e-3
n_of_x = lambda x: trench_density(x, L, n_bg, n_floor, half_width, ramp)
dt = pick_dt(L / Nx, Bx0, n_bg, c_eff, n0, p, n_floor_eps)

sim = Sim(Nx, L, Bx0, n_of_x, n0, p, c_eff, n_floor_eps, dt, use_blend=True)

xc = 0.5 * L
x0 = xc - half_width - 8.0
k_mode = 2 * np.pi * 3 / (2 * (half_width + 8.0))
sim.By[:] = 1e-3 * gaussian_packet(sim.xB, x0, 2.0, k_mode)
sim.Bz[:] = 0.0

incident_amp = np.max(np.abs(sim.By))
vg_estimate = 2 * k_mode * Bx0 / n_bg
distance_to_clear_trench = (xc + half_width) - x0
t_needed = distance_to_clear_trench / vg_estimate
nsteps = int(1.3 * t_needed / dt)
print(f"k_mode={k_mode:.4f} vg~{vg_estimate:.3f} dt={dt:.3e} nsteps={nsteps}")

maxfield_history = []
snapshots_C = []
snap_times_C = []
record_every = max(1, nsteps // 60)
t0 = time.time()
for it in range(nsteps):
    sim.step()
    if it % 500 == 0:
        maxfield_history.append(max(np.max(np.abs(sim.By)), np.max(np.abs(sim.Bz))))
    if it % record_every == 0:
        snapshots_C.append((sim.By.copy(), sim.Bz.copy()))
        snap_times_C.append(it * dt)
print(f"  ({time.time()-t0:.1f}s wall)")

maxfield_history = np.array(maxfield_history)
print(f"max|field| over run: min={maxfield_history.min():.3e} max={maxfield_history.max():.3e}")
assert np.all(np.isfinite(maxfield_history)), "fields must stay finite through the vacuum trench"
assert maxfield_history.max() < 100 * incident_amp, "no blow-up: fields should stay O(incident amplitude)"
print("PASS (bounded, no blow-up)")

left_mask = sim.xB < xc
right_mask = sim.xB >= xc
E_left = np.sum(sim.energy()[left_mask])
E_right = np.sum(sim.energy()[right_mask])
print(f"final energy: left half = {E_left:.3e}, right half = {E_right:.3e}, "
     f"transmitted-side fraction ~ {E_right/(E_left+E_right):.3f}")
results['C'] = dict(maxfield_history=maxfield_history, snapshots=snapshots_C, snap_times=snap_times_C,
                    xE=sim.xE, xB=sim.xB, n=sim.n, incident_amp=incident_amp,
                    E_left=E_left, E_right=E_right, xc=xc, half_width=half_width)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenario_results.npz")
np.savez(out_path, results=results, allow_pickle=True)
print(f"\nAll scenarios finished, results saved to {out_path}.")
