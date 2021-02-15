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



def config():

    # configure the simulation
    # most unstable mode at k=0.19, that is lambda = 33
    # hence only the fundamental mode in the box

    Simulation(
        smallest_patch_size=20,
        largest_patch_size=20,
        time_step_nbr=10,          # number of time steps (not specified if time_step and final_time provided)
        time_step=0.01,            # simulation final time (not specified if time_step and time_step_nbr provided)
        boundary_types="periodic", # boundary condition, string or tuple, length == len(cell) == len(dl)
        cells=165,                 # integer or tuple length == dimension
        dl=0.2,                    # mesh size of the root level, float or tuple
        refinement_boxes={"L0": {"B0": [(40, ), (120, )]},
                          "L1": {"B0": [(100,), (200,)]}},
        diag_options={"format": "phareh5", "options": {"dir": "2stream","mode":"overwrite"}}
    )


    def densityMain(x):
        return 1.

    def densityBeam(x):
        return .01

    def bx(x):
        return 1.

    def by(x):
        return 0.

    def bz(x):
        return 0.

    def vB(x):
        return 0.

    def v0(x):
        return 0.

    def vth(x):
        # thermal velocity for T=0.1
        return np.sqrt(2*0.1)


    vMain = {
        "vbulkx": v0, "vbulky": v0, "vbulkz": v0,
        "vthx": vth, "vthy": vth, "vthz": vth
    }


    vBulk = {
        "vbulkx": vB, "vbulky": v0, "vbulkz": v0,
        "vthx": vth, "vthy": vth, "vthz": vth
    }


    MaxwellianFluidModel(
        bx=bx, by=by, bz=bz,
        main={"charge": 1, "density": densityMain, **vMain, "init": {"seed": 1337}},
        beam={"charge": 1, "density": densityBeam, **vBulk, "init": {"seed": 1337}}
    )


    ElectronModel(closure="isothermal", Te=np.sqrt(2*0.1))


    sim = ph.global_vars.sim

    timestamps = np.arange(0, sim.final_time +sim.time_step, 2*sim.time_step)


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



def plot(bhier):
    times = np.sort(np.asarray(list(bhier.time_hier.keys())))

    components  =("B_x", "B_y", "B_z")
    ylims = ((0,1.5),(-0.25,0.25),(-0.25,0.25))

    for component,ylim in zip(components,ylims):
        for it,t in enumerate(times):
            fig,ax = plt.subplots(figsize=(10,6))
            for il,level in bhier.levels(t).items():
                patches = level.patches
                if il == 0:
                    marker="+"
                    alpha=1
                    ls='-'
                else:
                    marker='o'
                    alpha=0.4
                    ls='none'

                for ip, patch in enumerate(patches):
                    val   = patch.patch_datas["EM_"+component].dataset[:]
                    x_val = patch.patch_datas["EM_"+component].x
                    label="${}$ level {} patch {}".format(component,il,ip)
                    ax.plot(x_val, val, label=label,
                            marker=marker, alpha=alpha, ls=ls)
                    ax.set_ylim(ylim)

            ax.legend(ncol=4)
            ax.set_title("t = {:05.2f}".format(t))
            fig.savefig("{}_{:04d}.png".format(component,it))
            plt.close(fig)




def main():
    config()
    simulator = Simulator(gv.sim)
    simulator.initialize()
    simulator.run()


    #if cpp.mpi_rank() == 0:
    #    b = hierarchy_from(h5_filename="phare_outputs/EM_B.h5")
    #    plot(b)
    #
if __name__=="__main__":
    main()
