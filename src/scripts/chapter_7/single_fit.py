import paths
import numpy as np
import starry
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import arviz as az

np.random.seed(42)

starry.config.lazy = False
starry.config.quiet = True


# Load data
fsim = np.load(paths.data/"output/mapping_exo/T341_1bar_fluxes/hd209/fsim_snapshots.npy")[0]
fsim_ref = np.load(paths.data/"output/mapping_exo/T341_1bar_fluxes/hd209/fsim_reference_snapshots.npy")[0]
fobs1, ferr1 = np.load(
   paths.data/"output/mapping_exo/T341_1bar_mcmc/hd209/time_0/snr_15/lightcurve.npy"
)
fobs2, ferr2 = np.load(
   paths.data/"output/mapping_exo/T341_1bar_mcmc/hd209/time_0/snr_50/lightcurve.npy"
)

t_eclipse = np.load(paths.data/"output/mapping_exo/T341_1bar_fluxes/hd209/t_eclipse.npy")

# Normalize
norm = np.max(fsim_ref)
fsim_ref /= norm
fsim /= norm

l = 4
samples1 = az.from_netcdf(
   paths.data/f"output/mapping_exo/T341_1bar_mcmc/hd209/time_0/snr_15/l_{l}/samples.nc"
)
samples2 = az.from_netcdf(
   paths.data/f"output/mapping_exo/T341_1bar_mcmc/hd209/time_0/snr_50/l_{l}/samples.nc"
)
map_sim_temp = np.load(
    paths.data/"output/mapping_exo/T341_1bar_mcmc/hd209/time_0/sim_map_temp.npy",
)
dat = np.load(
   paths.data/f"output/mapping_exo/T341_1bar_mcmc/hd209/time_0/snr_15/l_{l}/map_inf_temp.npz"
) 
map_inf_temp_mean1 = dat["mean"]
map_inf_temp_samples1 = dat["samples"]

dat = np.load(
   paths.data/f"output/mapping_exo/T341_1bar_mcmc/hd209/time_0/snr_50/l_{l}/map_inf_temp.npz"
) 
map_inf_temp_mean2 = dat["mean"]
map_inf_temp_samples2 = dat["samples"]

fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(
    nrows=7,
    ncols=3 + 1 + 3 + 1 + 3 + 1 + 3,
    width_ratios=[1,1,1,0.2,1,1,1,0.2,1,1,1,2.,1,1,2.],
)

# Axes for simulated maps
ax_sim_maps = [
    fig.add_subplot(gs[0:3, 0:3]),
    fig.add_subplot(gs[4:7, 0:3]),
]

# Axes for inferred maps
ax_inf_maps = [
    fig.add_subplot(gs[0:3, 4:7]),
    fig.add_subplot(gs[4:7, 4:7]),
]

# Axes for inferred map samples
ax_samples1 = [fig.add_subplot(gs[i, 8 + j]) for i in range(3) for j in range(3)]
ax_samples2 = [fig.add_subplot(gs[4 + i, 8 + j]) for i in range(3) for j in range(3)]

# Axes for light curve fits
ax_lcs = [
    fig.add_subplot(gs[0:3, 12:15]),
    fig.add_subplot(gs[4:7, 12:15]),
]
vmin, vmax = 1400, 1800
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = "turbo"

# Plot the simulated maps
map = starry.Map(10)
for a in ax_sim_maps:
   map.show(
    image=map_sim_temp, projection="orthogonal", cmap=cmap, ax=a, norm=norm, zorder=-1
)

# Plot inferred mean maps
map.show(
    image=map_inf_temp_mean1, projection="orthogonal", cmap=cmap, ax=ax_inf_maps[0], norm=norm, zorder=-1
)
map.show(
    image=map_inf_temp_mean2, projection="orthogonal", cmap=cmap, ax=ax_inf_maps[1], norm=norm, zorder=-1
)

# Plot inferred map samples
for i in range(9):
   map.show(
    image=map_inf_temp_samples1[i], projection="orthogonal", cmap=cmap, ax=ax_samples1[i], norm=norm, zorder=-1,
        grid=False,
    )
   map.show(
    image=map_inf_temp_samples2[i], projection="orthogonal", cmap=cmap, ax=ax_samples2[i], norm=norm, zorder=-1,
        grid=False,
   )

# Plot flux
fpred1_median = samples1.posterior.isel(chain=0).fpred.median(dim="draw").values
fpred2_median = samples2.posterior.isel(chain=0).fpred.median(dim="draw").values


ax_lcs[0].plot(t_eclipse/3.5, (fobs1 - 1)*1e06, 'ko', alpha=0.03)
ax_lcs[1].plot(t_eclipse/3.5, (fobs2 - 1)*1e06, 'ko', alpha=0.03)
ax_lcs[0].plot(t_eclipse/3.5, (fpred1_median - 1)*1e06, 'C0-', lw=1.5)
ax_lcs[1].plot(t_eclipse/3.5, (fpred2_median - 1)*1e06, 'C0-', lw=1.5)

ax_sim_maps[0].set_title("Simulation\nsnapshot\n")
ax_inf_maps[0].set_title("Inferred\nmap\n")
ax_samples1[1].set_title("Inferred map\nsamples\n")

# Colorbar
cax = fig.add_axes([0.25, 0.02, 0.3, 0.02])
fig.colorbar(
    mpl.cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        cax=cax,
            orientation='horizontal',
            label="blackbody temperature [K]", fraction=1.
    )

ax_lcs[0].set_title("Predicted flux\n")

for a in ax_lcs:
    a.set_ylim(-1700, 500)
    a.grid(alpha=0.5)
    a.set_xlabel('phase') 
    a.set_ylabel("$(F/F_\mathrm{max} - 1)\\times 10^6$")
    a.set(xlim=(-0.025, 0.025), ylim=(-1700, 500), yticks=([-1500, -1000, -500, 0, 500]))
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())

# add text in ax_sim_maps[0] coordinates
ax_sim_maps[0].text(
    -0.25,
    0.2,
    "SNR = 15",
    transform=ax_sim_maps[0].transAxes,
    rotation=90,
    fontweight="bold",
    fontsize=16,
)

ax_sim_maps[1].text(
    -0.25,
    0.2,
    "SNR = 50",
    transform=ax_sim_maps[1].transAxes,
    rotation=90,
    fontweight="bold",
    fontsize=16,
)

for a in ax_sim_maps + ax_inf_maps + ax_samples1 + ax_samples2 + ax_lcs:
    a.set_rasterization_zorder(0)


fig.savefig(paths.figures/"single_fit.pdf",  bbox_inches='tight')