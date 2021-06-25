#!/usr/bin/env python3

import pyphare.pharein as ph #lgtm [py/import-and-import-from]
from pyphare.pharein import Simulation
from pyphare.pharein import MaxwellianFluidModel
from pyphare.pharein import ElectromagDiagnostics, FluidDiagnostics
from pyphare.pharein import ElectronModel
from pyphare.simulator.simulator import Simulator
from pyphare.pharein import global_vars as gv

from pyphare.pharesee.hierarchy import finest_field
import os
from pyphare.pharesee.run import Run
from pyphare.pharesee.hierarchy import get_times_from_h5

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.use('Agg')




def fromNoise():

    # in this configuration there are no prescribed waves
    # and only eigen modes existing in the simulation noise
    # will be visible

    Simulation(
        smallest_patch_size=20,
        largest_patch_size=20,

        # the following time step number
        # and final time mean that the
        # smallest frequency will be 2/100
        # and the largest 2/dt  = 2e3
        time_step_nbr=100000,
        final_time=100.,

        boundary_types="periodic",

        # smallest wavelength will be 2*0.2=0.4
        # and largest 50
        cells=500,
        dl=0.2,
        diag_options={"format": "phareh5",
                      "options": {"dir": "dispersion",
                                  "mode":"overwrite"}}
    )

    def density(x):
        return 1.


    def by(x):
        return 0.


    def bz(x):
        return 0.


    def bx(x):
        return 1.


    def vx(x):
        return 0.


    def vy(x):
        return 0.

    def vz(x):
        return 0.


    def vthx(x):
        return 0.01


    def vthy(x):
        return 0.01


    def vthz(x):
        return 0.01


    vvv = {
        "vbulkx": vx, "vbulky": vy, "vbulkz": vz,
        "vthx": vthx, "vthy": vthy, "vthz": vthz
    }

    MaxwellianFluidModel(
        bx=bx, by=by, bz=bz,
        protons={"charge": 1, "density": density, **vvv}
    )

    ElectronModel(closure="isothermal", Te=0.)


    sim = ph.global_vars.sim

    timestamps = np.arange(0, sim.final_time +sim.time_step, sim.time_step)

    for quantity in ["E", "B"]:
        ElectromagDiagnostics(
            quantity=quantity,
            write_timestamps=timestamps,
            compute_timestamps=timestamps,
        )




def omega(k, p):
    k2 = k*k
    return 0.5*k2*(np.sqrt(1+4/k2)+p)




def setOfModes():

    Simulation(
        smallest_patch_size=20,
        largest_patch_size=50,

        # smallest frequency is 0.06 (2pi/Tmax)
        # largest frequency is 3140 (2pi/dt)
        time_step_nbr=20000,
        final_time=20.,

        boundary_types="periodic",

        # smallest wavelength is 0.8 = 4 grid pts
        # largest wavelength is 50 = the whole domain (250 pts)
        cells=500,
        dl=0.2,
        diag_options={"format": "phareh5",
                      "options": {"dir": "dispersion1d",
                                  "mode":"overwrite"}}
    )


    # list of modes : m = 1 is for 1 wavelength in the whole domain
    modes = [4, 8, 16, 32, 64]

    # lists of amplitudes of the magnetic field amplitudes
    b_amplitudes = [0.1, 0.1, 0.1, 0.1, 0.2]

    # list of polarization : +1 for R mode and -1 for L mode
    polarizations = [+1, +1, +1, +1, +1]

    # list of phase at origin for magnetic and velocity fluctuations
    phases = [0 , 0, 0, 0, 0]

    assert(len(modes) == len(b_amplitudes) == len(polarizations) == len(phases))

    # list of wave_numbers for the given box
    from pyphare.pharein.global_vars import sim
    L = sim.simulation_domain()[0]
    wave_numbers = [2*np.pi*m/L for m in modes]

    # using faraday : v1 = -w b1 / (k . B0)
    v_amplitudes = [-b*omega(k, p)/k for (k, b, p) in zip(wave_numbers, b_amplitudes, polarizations)]


    def density(x):
        # no density fluctuations as whistler and AIC are not compressional
        return 1.


    def by(x):
        modes = 0.0
        for (k, b, f) in zip(wave_numbers, b_amplitudes, phases):
            modes += b*np.cos(k*x+f)
        return modes


    def bz(x):
        modes = 0.0
        for (k, b, f) in zip(wave_numbers, b_amplitudes, phases):
            modes += b*np.sin(k*x+f)
        return modes


    def bx(x):
        return 1.


    def vx(x):
        return 0.


    def vy(x):
        modes = 0.0
        for (k, v, f) in zip(wave_numbers, v_amplitudes, phases):
            modes += v*np.cos(k*x+f)
        #return modes
        return 0.0


    def vz(x):
        modes = 0.0
        for (k, v, f) in zip(wave_numbers, b_amplitudes, phases):
            modes += v*np.sin(k*x+f)
        #return modes
        return 0.0


    def vthx(x):
        return 0.01


    def vthy(x):
        return 0.01


    def vthz(x):
        return 0.01


    vvv = {
        "vbulkx": vx, "vbulky": vy, "vbulkz": vz,
        "vthx": vthx, "vthy": vthy, "vthz": vthz
    }


    MaxwellianFluidModel(
        bx=bx, by=by, bz=bz,
        main={"charge": 1, "density": density, **vvv}
    )

    ElectronModel(closure="isothermal", Te=0.)

    sim = ph.global_vars.sim

    timestamps = np.arange(0, sim.final_time+sim.time_step, 40*sim.time_step)


    for quantity in ["E", "B"]:
        ElectromagDiagnostics(
            quantity=quantity,
            write_timestamps=timestamps,
            compute_timestamps=timestamps,
        )


    for quantity in ["density", "bulkVelocity"]:
        FluidDiagnostics(
            quantity=quantity,
            write_timestamps=timestamps,
            compute_timestamps=timestamps,
            )

    return wave_numbers, v_amplitudes, b_amplitudes



# ___ post-processing functions
def get_all_w(run_path, wave_numbers):
    file = os.path.join(run_path, "EM_B.h5")
    times = get_times_from_h5(file)

    nm = len(wave_numbers)
    print(nm)

    r = Run(run_path)
    byz = np.array([])

    print(times.shape)
    for time in times:
        #print("time : ", time)
        B = r.GetB(time)
        by, x = finest_field(B, "By")
        bz, x = finest_field(B, "Bz")

        # x_fine = np.arange(x[0], x[-1], 0.05) # last arg : smallest grid size
        # by_fine = np.interp(x_fine, x, by)
        # bz_fine = np.interp(x_fine, x, bz)
        time_sample = by+1j*bz
        byz = np.concatenate((byz, time_sample))

    nx = x.shape[0]
    nt = times.shape[0]
    print(by.shape)
    print(byz.shape)
    byz = np.reshape(byz, (times.shape[0], x.shape[0]))
    #byz = np.reshape(byz, (x.shape[0], times.shape[0]))
    print(byz.shape)

    zob = np.absolute(np.fft.fft2(byz)[:(nt+1)//2, :(nx+1)//2])
    #zob = np.absolute(np.fft.fft2(byz)[:(nx+1)//2, :(nt+1)//2])
    print(zob.shape)
    print(dir(zob))

    idx = np.unravel_index(np.argmax(zob, axis=None), zob.shape)
    print(idx, zob[idx])



    # xv, yv = np.meshgrid()
    # idx=argsort(zob.flatten)
    # idx.reshape(,)
    # km = xv[idx[:n]], wm = yv[idx[:n]]



    return np.zeros_like(wave_numbers), zob



def main():
    wave_nums, v1, b1 = setOfModes()
    simulator = Simulator(gv.sim)
    simulator.initialize()
    simulator.run()

    from pybindlibs.cpp import mpi_rank

    if mpi_rank() == 0:
        omegas, zobi = get_all_w(os.path.join(os.curdir, "dispersion1d"), wave_nums)

        print(*('k = {:.4f}   w = {:.4f}   v = {:.4f}   b = {:.4f}'.format(k, w, v, b) for (k, w, v, b) in zip(wave_nums, omegas, v1, b1)), sep="\n")

        fig, ax = plt.subplots(figsize=(6,4), nrows=1)

        ax.imshow(zobi, origin='lower', cmap='viridis_r')

        fig.tight_layout()
        fig.savefig("dispersion.pdf", dpi=200)


        assert(1 == 1)


if __name__=="__main__":
    main()
