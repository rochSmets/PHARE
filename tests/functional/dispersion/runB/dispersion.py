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
                      "options": {"dir": "fromNoise1d",
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




def setOfModes(polarization, modes, b_amplitudes):

    Simulation(
        smallest_patch_size=20,
        largest_patch_size=50,

        # smallest frequency is 0.06 (2pi/Tmax)
        # largest frequency is 3140 (2pi/dt)
        time_step_nbr=80000,
        final_time=80.,

        boundary_types="periodic",

        # smallest wavelength is 0.8 = 4 grid pts
        # largest wavelength is 50 = the whole domain (250 pts)
        cells=4000,
        dl=0.2,
        diag_options={"format": "phareh5",
                      "options": {"dir": "setOfModes1d",
                                  "mode":"overwrite"}}
    )

    assert(len(modes) == len(b_amplitudes))

    # list of wave_numbers for the given box
    from pyphare.pharein.global_vars import sim
    L = sim.simulation_domain()[0]
    wave_numbers = [2*np.pi*m/L for m in modes]

    # using faraday : v1 = -w b1 / (k . B0)
    #v_amplitudes = [-b*omega(k, p)/k for (k, b, p) in zip(wave_numbers, b_amplitudes, polarizations)]


    def density(x):
        # no density fluctuations as whistler and AIC are not compressional
        return 1.

    def by(x):
        modes = 0.0
        for (k, b) in zip(wave_numbers, b_amplitudes):
            modes += b*np.cos(k*x)
        return modes

    def bz(x):
        modes = 0.0
        for (k, b) in zip(wave_numbers, b_amplitudes):
            modes += b*np.sin(k*x)*polarization
        return modes

    def bx(x):
        return 1.

    def vx(x):
        return 0.

    def vy(x):
        #modes = 0.0
        #for (k, v, f) in zip(wave_numbers, v_amplitudes, phases):
        #    modes += v*np.cos(k*x+f)
        #return modes
        return 0.0

    def vz(x):
        #modes = 0.0
        #for (k, v, f) in zip(wave_numbers, b_amplitudes, phases):
        #    modes += v*np.sin(k*x+f)
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

    return wave_numbers, b_amplitudes



# ___ post-processing functions
def get_all_w(run_path, wave_numbers, polarization):
    file = os.path.join(run_path, "EM_B.h5")
    times = get_times_from_h5(file)

    nm = len(wave_numbers)
    print('number of modes : {}'.format(nm))

    r = Run(run_path)
    byz = np.array([])

    for time in times:
        B = r.GetB(time)
        by, x = finest_field(B, "By")
        bz, x = finest_field(B, "Bz")

        # polarization = +1 for R mode, -1 for L mode
        byz = np.concatenate((byz, by+polarization*1j*bz))

    nx = x.shape[0]
    nt = times.shape[0]
    byz = np.reshape(byz, (nt, nx))

    BYZ = np.absolute(np.fft.fft2(byz)[:(nt+1)//2, :(nx+1)//2])
    BYZ_4_all_W = np.sum(BYZ, axis=0)

    idx = np.argsort(BYZ_4_all_W)
    kmodes = idx[-nm:]

    wmodes = []
    for i in range(nm):
        wmodes.append(np.argmax(BYZ[:,idx[-nm+i]]))

    print(kmodes, wmodes)

    return kmodes, np.array(wmodes), BYZ



def main():
    # list of modes : m = 1 is for 1 wavelength in the whole domain
    modes = [4, 8, 16, 32, 64, 128, 256, 512]

    # lists of amplitudes of the magnetic field amplitudes
    b_amplitudes = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    # polarization : -1 for L mode
    wave_nums, b1 = setOfModes(-1, modes, b_amplitudes)
    simulator = Simulator(gv.sim)
    simulator.initialize()
    simulator.run()

    from pybindlibs.cpp import mpi_rank
    from matplotlib import rc

    if mpi_rank() == 0:
        sim = ph.global_vars.sim

        L = sim.simulation_domain()[0]
        T = sim.final_time

        #for the left mode
        ki, wi, zobi = get_all_w(os.path.join(os.curdir, "setOfModes1d"), wave_nums, -1)

        k_numL = 2*np.pi*ki/L
        w_numL = 2*np.pi*wi/T
        print(*('Left mode... k = {:.4f}   w = {:.4f}   b = {:.4f}'.format(k, w, b) for (k, w, b) in zip(k_numL, w_numL, b1)), sep="\n")

    ph.global_vars.sim = None

    # list of modes : m = 1 is for 1 wavelength in the whole domain
    modes = [4, 8, 16, 32, 64, 128, 256, 512]

    # lists of amplitudes of the magnetic field amplitudes
    b_amplitudes = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    # polarization : +1 for R mode
    wave_nums, b1 = setOfModes(+1, modes, b_amplitudes)
    simulator = Simulator(gv.sim)
    simulator.initialize()
    simulator.run()

    if mpi_rank() == 0:
        #sim = ph.global_vars.sim

        L = sim.simulation_domain()[0]
        T = sim.final_time

        #for the riht mode
        ki, wi, zobi = get_all_w(os.path.join(os.curdir, "setOfModes1d"), wave_nums, +1)

        k_numR = 2*np.pi*ki/L
        w_numR = 2*np.pi*wi/T
        print(*('Right mode... k = {:.4f}   w = {:.4f}   b = {:.4f}'.format(k, w, b) for (k, w, b) in zip(k_numR, w_numR, b1)), sep="\n")

        rc('text', usetex = True)

        fig, ax = plt.subplots(figsize=(4,3), nrows=1)

        k_the = np.arange(0.04, 8, 0.001)
        w_thR = omega(k_the, +1)
        w_thL = omega(k_the, -1)

        #ax.imshow(zobi, origin='lower', cmap='viridis_r')
        ax.plot(k_the, w_thR, '-k')
        ax.plot(k_the, w_thL, '-k')
        ax.plot(k_numR, w_numR, 'b+', label='R mode', markersize=8)
        ax.plot(k_numL, w_numL, 'rx', label='L mode', markersize=8)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$k c / \omega_p$')
        ax.set_ylabel('$\omega / \Omega_p$')

        ax.legend(loc='upper left', frameon=False)

        fig.tight_layout()
        fig.savefig("dispersion.pdf", dpi=200)

        w_theR = omega(k_numR, +1)
        w_theL = omega(k_numL, -1)

        errorL = 100*np.fabs(w_numL-w_theL)/w_theL
        errorR = 100*np.fabs(w_numR-w_theR)/w_theR

        print('error L :', errorL)
        print('error R :', errorR)

        assert(1 == 1)


if __name__=="__main__":
    main()

