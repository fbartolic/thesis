import paths
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator


input_dir = paths.data/"output/mapping_exo/T341_1bar_fluxes/hd209/"
fsim = np.load(os.path.join(input_dir, "fsim_snapshots.npy"))
fsim_ref = np.load(os.path.join(input_dir, "fsim_reference_snapshots.npy"))
t_eclipse = np.load(os.path.join(input_dir, "t_eclipse.npy"))

# Normalize
norm = np.max(fsim_ref, axis=1)
fsim_ref /= norm[:, None]
fsim /= norm[:, None]

# Rescale 
fsim = (fsim - 1)*1e06
fsim_ref = (fsim_ref - 1)*1e06

fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"wspace":0.4})

# Plot eclipses
fsim_median = np.median(fsim, axis=0)
colors = plt.cm.viridis(np.linspace(0, 1, len(fsim)))

for fsim_, fsim_ref_ in zip(fsim, fsim_ref):
    ax[0].plot(t_eclipse/3.5, fsim_ - fsim_median, color="k", alpha=0.3, lw=0.8)
for fsim_, fsim_ref_ in zip(fsim, fsim_ref):
    ax[1].plot(t_eclipse/3.5, fsim_ - fsim_ref_, color="k", alpha=0.3, lw=0.8)

for a in ax:
    a.set_xlabel("Orbital phase")
    a.grid(alpha=0.5)
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())
#    a.set(xlim=(-0.25, 0.25))
    a.set(xlim=(-0.06, 0.06))

ax[0].set(ylim=(-70, 70))
ax[0].set(ylabel=r"$F - F_\mathrm{median}$ [ppm]")
ax[1].set(ylabel=r"$F_{l=20} - F_{l=2}$ [ppm]")
ax[0].set_title("Secondary eclipse variability between orbits")
ax[1].set_title("Deviation from baseline map (an $l=2$ map)")
ax[0].set_yticks(np.arange(-50, 75, 25))
ax[1].set_yticks(np.arange(-10, 15, 5));


fig.savefig(paths.figures/"flux_variation.pdf", bbox_inches="tight")
