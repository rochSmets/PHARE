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




def setOfModes():

    wave_numbers = [0.125, 8.0] # 8, 4, 2, 1, 0.5, 0.25, 0.125
    b_amplitudes = [0.1  , 0.1]

    Simulation(
        smallest_patch_size=20,
        largest_patch_size=50,

        # smallest frequency is 0.06 (2pi/Tmax)
        # largest frequency is 3140 (2pi/dt)
        time_step_nbr=100,
        #time_step_nbr=100000,
        final_time=100.,

        boundary_types="periodic",

        # smallest wavelength is 0.8 = 4 grid pts
        # largest wavelength is 50 = the whole domain (250 pts)
        cells=250,
        dl=0.2,
        diag_options={"format": "phareh5",
                      "options": {"dir": "dispersion",
                                  "mode":"overwrite"}}
    )

    def density(x):
        # no density fluctuations as whistler and AIC are not compressional
        return 1.


    def by(x):
        from pyphare.pharein.global_vars import sim
        L = sim.simulation_domain()
        return 0.1*np.cos(2*np.pi*x/L[0])


    def bz(x):
        from pyphare.pharein.global_vars import sim
        L = sim.simulation_domain()
        return -0.1*np.sin(2*np.pi*x/L[0])


    def bx(x):
        return 1.


    def vx(x):
        return 0.


    def vy(x):
        from pyphare.pharein.global_vars import sim
        L = sim.simulation_domain()
        return 0.1*np.cos(2*np.pi*x/L[0])

    def vz(x):
        from pyphare.pharein.global_vars import sim
        L = sim.simulation_domain()
        return 0.1*np.sin(2*np.pi*x/L[0])


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
