#!/usr/bin/env python3

import pyphare.pharein as ph #lgtm [py/import-and-import-from]
from pyphare.pharein import Simulation
from pyphare.pharein import MaxwellianFluidModel
from pyphare.pharein import ElectromagDiagnostics, FluidDiagnostics
from pyphare.pharein import ElectronModel
from pyphare.simulator.simulator import Simulator
from pyphare.pharein import global_vars as gv


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
        time_step_nbr=100,
        final_time=1.,

        boundary_types="periodic",

        # smallest wavelength is 0.8 = 4 grid pts
        # largest wavelength is 50 = the whole domain (250 pts)
        cells=250,
        dl=0.2,
        diag_options={"format": "phareh5",
                      "options": {"dir": "dispersion",
                                  "mode":"overwrite"}}
    )


    # list of modes : m = 1 is for 1 wavelength in the whole domain
    modes = [1, 64]

    # lists of amplitudes of the magnetic field amplitudes
    b_amplitudes = [0.1, 0.1]

    # list of polarization : +1 for R mode and -1 for L mode
    polarizations = [+1, +1]

    # list of phase at origin for magnetic and velocity fluctuations
    phases = [0 , 0]

    assert(len(modes) == len(b_amplitudes) == len(polarizations) == len(phases))

    # list of wave_numbers for the given box
    from pyphare.pharein.global_vars import sim
    L = sim.simulation_domain()[0]
    wave_numbers = [2*np.pi*m/L for m in modes]
    print("wave_numbers : ", wave_numbers)

    # using faraday : v1 = -w b1 / (k . B0)
    v_amplitudes = [-b*omega(k, p)/k for (k, b, p) in zip(wave_numbers, b_amplitudes, polarizations)]
    print("v_amplitudes", v_amplitudes)


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
        return modes


    def vz(x):
        modes = 0.0
        for (k, v, f) in zip(wave_numbers, b_amplitudes, phases):
            modes += v*np.sin(k*x+f)
        return modes


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

    timestamps = np.arange(0, sim.final_time +sim.time_step, sim.time_step)


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




def main():
    setOfModes()
    simulator = Simulator(gv.sim)
    simulator.initialize()
    simulator.run()

if __name__=="__main__":
    main()
