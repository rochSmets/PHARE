"""
1D toy prototype validating the density-dependent relaxation blend between
Ohm's law and vacuum Ampere's law for E (design doc:
doc/design/vacuum_ampere_ohm_blend.md, step (b) of section 11).

Model (see design doc section 4 for the physics, and the doc's addenda for
numerical caveats):

  - 1D periodic domain, x in [0, L). Background field Bx0 along x (constant,
    unevolved -- matches PHARE's 1D Faraday/Ampere, which leave Bx untouched).
  - Transverse fields By, Bz (Yee half-integer grid) and Ey, Ez (Yee integer
    grid), evolved via Faraday's law:  dBy/dt = dEz/dx ,  dBz/dt = -dEy/dx.
  - J from the static (no-displacement-current) Ampere's law:
        Jy = -dBz/dx ,  Jz = dBy/dx
  - Ohm's law, Hall-term only (immobile ions Vi=0, so Ve=-J/(n_floored e),
    no electron pressure -- the minimal model that reproduces the 1/n
    singularity in the real ohm.hpp that this whole feature exists to
    regularize) plus a *small* hyper-resistive term (-nu*laplacian(J),
    same form as ohm.hpp's constant_hyperresistive_): see the "on time
    integration" note below for why this term turned out to be load
    bearing here, not just a physics nicety:
        E_Ohm_y =  Jz * Bx0 / n_floored - nu*laplacian(Jy)
        E_Ohm_z = -Jy * Bx0 / n_floored - nu*laplacian(Jz)
  - E is blended via the exact exponential-integrator relaxation from the
    design doc:
        g = (n / n0)^p ,  w = exp(-g) ,  h(g) = (1-exp(-g))/g
        E_new = w*E_old + (1-w)*E_Ohm + dt*h(g)*c_eff^2*curl(B)
    with curl(B) = J (same current used for the Hall term).

On time integration: fields are advanced with standard single-level Yee
time-staggering (B first, from the *current* E; then E, from the *new*
B) -- not the naive "compute E fully then B fully" ordering a first
version of this script used. That ordering turns out to matter a lot:
- The naive/"either order, single-level" scheme is *unconditionally*,
  weakly unstable for the pure Hall/whistler rotation at every
  wavenumber (its single-mode transfer matrix has eigenvalues
  1+-i*Omega, |.| = sqrt(1+Omega^2) > 1 for any nonzero Omega,
  regardless of which field is advanced first) -- confirmed both
  empirically (slow-onset but real exponential blow-up; a high-accuracy
  exact-time-integration reference confirmed the *spatial* discretization
  itself is fine) and analytically.
- A genuine multi-time-level leapfrog for B fixes the Hall term exactly
  (verified: eigenvalue magnitude exactly 1 for any k, dt) but, paired
  with an Euler-integrated (memory-carrying) E for the vacuum term,
  introduces a spurious third numerical mode that is itself weakly
  unstable and corrupts the vacuum dispersion measurement.
- Standard single-level Yee staggering (used here) has neither problem:
  it has exactly as many eigenvalues as physical degrees of freedom (no
  spurious mode; vacuum dispersion matches analytic to ~0.01%), and
  reduces the Hall-term instability's growth rate enough that a small,
  physically-motivated hyper-resistive term (see above) fully controls
  it -- the same role this term plays in the real ohm.hpp.
- Separately: a too-sharp density gradient (e.g. a trench ramp resolved
  by too few cells) can trigger a genuine parametric numerical
  instability from the *spatially varying* coefficient, independent of
  the time-integration issues above. This does not show up in a
  spatially uniform test and was only caught by the trench scenario --
  see the ramp width chosen in the driver script.

This is a standalone validation script (numpy only), not wired to the real
PHARE solver. See the run_scenarios.py driver for what it checks.
"""

import numpy as np


# ---------------------------------------------------------------- geometry --
def make_grid(Nx, L):
    dx = L / Nx
    xE = np.arange(Nx) * dx           # E, J, n live here (integer nodes)
    xB = (np.arange(Nx) + 0.5) * dx   # B lives here (half nodes)
    return dx, xE, xB


def dEdx_at_B(E, dx):
    return (np.roll(E, -1) - E) / dx


def dBdx_at_E(B, dx):
    return (B - np.roll(B, 1)) / dx


def laplacian_at_E(F, dx):
    return (np.roll(F, -1) - 2.0 * F + np.roll(F, 1)) / dx ** 2


# --------------------------------------------------------------- densities --
def uniform_density(xE, n):
    return np.full_like(xE, n)


def trench_density(xE, L, n_bg, n_floor, half_width, ramp):
    xc = 0.5 * L
    trench = 0.5 * (np.tanh((xE - (xc - half_width)) / ramp)
                    - np.tanh((xE - (xc + half_width)) / ramp))
    return n_bg - (n_bg - n_floor) * trench


# ------------------------------------------------------------ blend pieces --
def g_of_n(n, n0, p):
    return np.clip(n, 0.0, None) ** p / n0 ** p


def w_of_g(g):
    return np.exp(-g)


def h_of_g(g):
    out = np.ones_like(g)
    mask = g > 1e-12
    out[mask] = -np.expm1(-g[mask]) / g[mask]
    return out


# -------------------------------------------------------------------- sim --
class Sim:
    """
    Standard single-level Yee time-staggering: B is advanced first (using
    the current E), then E is advanced using the just-updated B:
        B^{n+1/2} = B^{n-1/2} + dt*curl(E^n)
        E^{n+1}   = blend(E^n, E_Ohm(B^{n+1/2}), curl(B^{n+1/2}), n, dt)
    See the module docstring for why this specific ordering (and the small
    hyper-resistive term in ohm_hall) was chosen over two other schemes
    that were tried and found wanting.
    """

    def __init__(self, Nx, L, Bx0, n_of_x, n0, p, c_eff, n_floor_eps, dt, use_blend=True,
                nu=1e-4):
        self.dx, self.xE, self.xB = make_grid(Nx, L)
        self.Bx0 = Bx0
        self.n = n_of_x(self.xE)
        self.n0 = n0
        self.p = p
        self.c_eff = c_eff
        self.n_floor_eps = n_floor_eps
        self.dt = dt
        self.use_blend = use_blend
        self.nu = nu

        self.g = g_of_n(self.n, n0, p)
        self.w = w_of_g(self.g)
        self.h = h_of_g(self.g)
        self.n_floored = np.maximum(self.n, n_floor_eps)

        self.By = np.zeros(Nx)
        self.Bz = np.zeros(Nx)
        self.Ey = np.zeros(Nx)
        self.Ez = np.zeros(Nx)

    def ohm_hall(self, Jy, Jz):
        Ey_ohm = Jz * self.Bx0 / self.n_floored - self.nu * laplacian_at_E(Jy, self.dx)
        Ez_ohm = -Jy * self.Bx0 / self.n_floored - self.nu * laplacian_at_E(Jz, self.dx)
        return Ey_ohm, Ez_ohm

    def step(self):
        self.By = self.By + self.dt * dEdx_at_B(self.Ez, self.dx)
        self.Bz = self.Bz - self.dt * dEdx_at_B(self.Ey, self.dx)

        Jy = -dBdx_at_E(self.Bz, self.dx)
        Jz = dBdx_at_E(self.By, self.dx)

        Ey_ohm, Ez_ohm = self.ohm_hall(Jy, Jz)

        if self.use_blend:
            Ey_new = self.w * self.Ey + (1 - self.w) * Ey_ohm \
                + self.dt * self.h * self.c_eff ** 2 * Jy
            Ez_new = self.w * self.Ez + (1 - self.w) * Ez_ohm \
                + self.dt * self.h * self.c_eff ** 2 * Jz
        else:
            Ey_new, Ez_new = Ey_ohm, Ez_ohm

        self.Ey, self.Ez = Ey_new, Ez_new

    def run(self, nsteps, record_every=None):
        history = []
        for it in range(nsteps):
            self.step()
            if record_every and it % record_every == 0:
                history.append((self.By.copy(), self.Bz.copy(), self.Ey.copy(), self.Ez.copy()))
        return history

    def max_abs_field(self):
        return max(np.max(np.abs(self.By)), np.max(np.abs(self.Bz)),
                   np.max(np.abs(self.Ey)), np.max(np.abs(self.Ez)))

    def energy(self):
        return 0.5 * (self.By ** 2 + self.Bz ** 2 + self.Ey ** 2 + self.Ez ** 2)


def max_hall_rate(Bx0, n0, p, n_floor_eps, n_bg):
    """max over n in (0, n_bg] of (1-w(g(n))) * Bx0 / max(n, n_floor_eps).

    Counterintuitively this is NOT maximized deep in the dense region (where
    (1-w) -> 1 but Bx0/n is small) nor deep in vacuum (where Bx0/n is huge
    but (1-w) -> 0 fast enough to compensate, by construction -- section
    4.2 of the design doc). It peaks close to n0 itself, where (1-w) is
    already O(1) but Bx0/n is still large. This sets the explicit-stability
    CFL bound for the Hall/whistler term, and is *more* restrictive than a
    naive dense-region-only estimate would suggest.
    """
    ns = np.logspace(np.log10(n_floor_eps), np.log10(n_bg), 4000)
    g = g_of_n(ns, n0, p)
    w = w_of_g(g)
    ratio = (1 - w) * Bx0 / np.maximum(ns, n_floor_eps)
    return np.max(ratio)


def pick_dt(dx, Bx0, n_bg, c_eff, n0, p, n_floor_eps, safety=0.4):
    k_max = np.pi / dx
    dt_vacuum_bound = 2.0 / (c_eff * k_max)
    hall_rate = max_hall_rate(Bx0, n0, p, n_floor_eps, n_bg)
    dt_whistler_bound = 2.0 / (hall_rate * k_max ** 2)
    return safety * min(dt_vacuum_bound, dt_whistler_bound)
