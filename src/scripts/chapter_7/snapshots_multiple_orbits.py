import paths
import numpy as np
import starry
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl

from utils import starry_intensity_to_bbtemp

np.random.seed(42)

starry.config.lazy = False
starry.config.quiet = True

def initialize_map(ydeg, nw, x):
    map = starry.Map(ydeg, nw=nw)
    map[1:, :, :] = x[1:, :] / x[0]
    map.amp = x[0]
    return map


# Wavelength grid for starry map (should match filter range)
wavelength_grid = np.linspace(4.5 - 1.2, 4.5 + 1.2, 50)

xs_sim = np.load(paths.data/"output/mapping_exo/T341_1bar_ylms/coefficients.npy")

maps_sim_temp = []
for i in range(30):
    x_sim = xs_sim[i]
    map = initialize_map(20, len(wavelength_grid), x_sim)
    map_sim_temp = starry_intensity_to_bbtemp(
        map.render(res=200, projection="Orthogonal"), wavelength_grid
    ) 
    maps_sim_temp.append(map_sim_temp)

fig, ax = plt.subplots(3, 10, figsize=(12, 4),  gridspec_kw={'hspace':0.4})
ax = ax.reshape(-1)

vmin, vmax= 1550, 1750
for i in range(len(maps_sim_temp)):
    map.show(
        image=maps_sim_temp[i], projection="orthogonal", cmap="OrRd", 
        norm=colors.Normalize(vmin=vmin, vmax=vmax),
        ax=ax[i],
        zorder=-1,
    )
    ax[i].set_aspect(1)
    ax[i].set_title(f"$t/P = {i}$")

# Colorbar
cax = fig.add_axes([0.36, 0.02, 0.3, 0.03])
fig.colorbar(
mpl.cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap="OrRd"),
    cax=cax,
        orientation='horizontal',
        label="blackbody temperature [K]", fraction=1.
)

for a in ax:
    a.set_rasterization_zorder(0)

fig.savefig(paths.figures/"snapshots_multiple_orbits.pdf", bbox_inches="tight")