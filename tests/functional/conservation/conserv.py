#!/usr/bin/env python3

import pyphare.pharein as ph #lgtm [py/import-and-import-from]
from pyphare.pharein import Simulation
from pyphare.pharein import MaxwellianFluidModel
from pyphare.pharein import ElectromagDiagnostics, FluidDiagnostics, ParticleDiagnostics
from pyphare.pharein import ElectronModel
from pyphare.simulator.simulator import Simulator
from pyphare.pharein import global_vars as gv


import os
import numpy as np
import pyphare
from pyphare.pharesee.run import Run
from pyphare.pharesee.hierarchy import get_times_from_h5
from glob import glob


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')


#############################################################
#
#                  SIMULATION CONFIG
#
#############################################################
def uniform(vth, dl, cells, nbr_steps):

    Simulation(
        smallest_patch_size=20,
        largest_patch_size=20,
        time_step_nbr=nbr_steps,
        final_time=50.,
        boundary_types="periodic",
        cells=cells,
        dl=dl,
        diag_options={"format": "phareh5",
                      "options": {"dir": "vth{}dx{}".format(vth,dl),
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
        return vth

    def vthy(x):
        return vth

    def vthz(x):
        return vth


    vvv = {
        "vbulkx": vx, "vbulky": vy, "vbulkz": vz,
        "vthx": vthx, "vthy": vthy, "vthz": vthz
    }

    MaxwellianFluidModel(
        bx=bx, by=by, bz=bz,
        protons={"charge": 1, "density": density, **vvv}
    )

    ElectronModel(closure="isothermal", Te=0.1)

    sim = ph.global_vars.sim

    timestamps = np.arange(0, sim.final_time, 50*sim.time_step)

    for quantity in ["B"]:
        ElectromagDiagnostics(
            quantity=quantity,
            write_timestamps=timestamps,
            compute_timestamps=timestamps,
        )

    for name in ["domain", "levelGhost", "patchGhost"]:
        ParticleDiagnostics(quantity=name,
                            compute_timestamps=timestamps,
                            write_timestamps=timestamps,
                            population_name="protons")



#############################################################
#               POSTPROCESS ROUTINES
#############################################################
#plotting et al. functions

def kinetic_energy(particles, kind="iso"):
    """
    return the total kinetic energy of given particles
    assuming parallel is x, and perp is y,z
    This also assumes protons, so mass is 1.
    """
    if kind=="iso":
        return 0.5*np.sum((particles.v[:,0]**2 +\
                           particles.v[:,1]**2 +\
                           particles.v[:,2]**2)* particles.weights)
    if kind=="perp":
        return 0.5*np.sum((particles.v[:,1]**2 +\
                           particles.v[:,2]**2)* particles.weights)
    if kind=="para":
        return 0.5*np.sum((particles.v[:,0]**2)* particles.weights)



def total_particles(parts, fun, lvlNbr=0, **kwargs):
    """
    given a particle hierarchy, calculate a total
    quantity on a given level, quantity being estimated
    by the callback "fun" (could be kinetic energy, momentum, et.)
    """
    for ilvl, lvl in parts.patch_levels.items():
        if lvlNbr == ilvl:
            tot = 0.
            for ip, patch in enumerate(lvl.patches):
                keys = list(patch.patch_datas.keys())
                pdata = patch.patch_datas[keys[0]]
                particles= pdata.dataset
                per_patch = fun(particles, **kwargs)
                tot += per_patch
    return tot



def total_kinetic(parts, lvlNbr=0, kind="iso"):
    return total_particles(parts, kinetic_energy, lvlNbr, kind=kind)



def mag_energy(B, lvlNbr=0):
    """
    return the total magnetic energy on a given level
    """
    for ilvl, lvl in B.patch_levels.items():
        if lvlNbr == ilvl:
            tot = 0.
            for ip, patch in enumerate(lvl.patches):
                pdata = patch.patch_datas["Bx"]

                # Bx is primal, but By and Bz dual so
                # we average By and Bz onto Bx to use them
                # in the same formula
                ghosts_nbr = pdata.ghosts_nbr[0]
                bx = patch.patch_datas["Bx"].dataset[ghosts_nbr:-ghosts_nbr]
                bytmp = patch.patch_datas["By"].dataset[ghosts_nbr-1:-(ghosts_nbr-1)]
                bztmp = patch.patch_datas["Bz"].dataset[ghosts_nbr-1:-(ghosts_nbr-1)]
                by = 0.5*(bytmp[1:]+bytmp[:-1])
                bz = 0.5*(bztmp[1:]+bztmp[:-1])

                # sum 0.5B^2 * dx over all nodes
                per_patch = np.sum((bx**2 + by**2 + bz**2)*0.5*pdata.layout.dl[0])
                tot +=  per_patch
        return tot


def energies(path, kkind="iso"):
    """
    This loops over all times of a given run
    and return the magnetic and kinetic energy
    as a function of time.
    """
    r = Run(path)
    times = get_times_from_h5(r.path+"/EM_B.h5")
    Bnrj = np.zeros_like(times)
    K = np.zeros_like(times)
    for it,t in enumerate(times):
        B = r.GetB(t)
        protons = r.GetParticles(t, "protons")
        Bnrj[it] = mag_energy(B)
        K[it] = total_kinetic(protons,kind=kkind)
    return r, Bnrj, K, times



def avg_interval(t1, t2, times):
    return [np.argmin(np.abs(times-t)) for t in (t1,t2)]




#############################################################
#############################################################



def main():

    cases = [0.01,0.05,0.1,0.3,0.5,0.75,1,2]

    dls = [0.2, 0.1]
    nbrcells = [100,200]
    nbrdts = [25000, 100000]

    for vth in cases:
        for dl, nbrcell, nbrdt in zip(dls, nbrcells, nbrdts):
            uniform(vth, dl, nbrcell, nbrdt)
            simulator = Simulator(gv.sim)
            simulator.initialize()
            simulator.run()
            gv.sim = None



    paths = glob("*vth*")
    runs_vth = {}
    Bnrj_vth = {}
    K_vth = {}
    times_vth={}

    #extract vth and dx from the name of the directory
    vthdx = np.asarray(sorted([[float(x) for x in path.split("/")[-1].strip("vth").split("dx")] for path in paths],
                   key=lambda x:x[1]))

    paths = sorted(paths, key=lambda path: float(paths[0].split("/")[-1].strip("vth").split("dx")[1]))

    #now for each directory, extract magnetic and kinetic energies
    for path in paths:
        runs_vth[path], Bnrj_vth[path], K_vth[path], times_vth[path] = energies(path)


    # we want to plot things as a function of the thermal velocity
    # for dx=0.1 and 0.2 so extract their values
    vth0p2 = np.asarray([x[0] for x in vthdx if x[1] == 0.2])
    vth0p1 = np.asarray([x[0] for x in vthdx if x[1] == 0.1])

    # we will plot the variation of the kinetic energy
    # relative to is "initial value", by "initial" we mean
    # its average over some time interval at the start of the run.
    # here we take 3,4 because before there is some kind of irrelevant transient
    K0 = {}
    for path,K in K_vth.items():
            it1, it2 = avg_interval(3,4, times_vth[path])
            K0[path] = np.mean(K[it1:it2+1])


    # calculate the relative variation of kinetic enery for both cases
    rel_K0p2 = np.asarray([np.abs(K[-1]-K0[path])/K0[path]*100 for path,K in K_vth.items() if "dx0.2" in path])
    rel_K0p1 = np.asarray([np.abs(K[-1]-K0[path])/K0[path]*100 for path,K in K_vth.items() if "dx0.1" in path])

    fig, ax = plt.subplots()
    id2 = np.argsort(vth0p2)
    id1 = np.argsort(vth0p1)
    ax.plot(vth0p2[id2], rel_K0p2[id2], marker="o", label="dx = 0.2 (dt=0.002, 25k steps)")
    ax.plot(vth0p1[id1], rel_K0p1[id1], marker="o", label="dx = 0.1 (dt=5e-4, 100k steps)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Delta K$ (%)")
    ax.set_xlabel("Vth")
    ax.set_title("kinetic energy evolution as a function of Vth")
    ax.legend()

    fig.tight_layout()
    fig.savefig("K.png")




if __name__=="__main__":
    main()
