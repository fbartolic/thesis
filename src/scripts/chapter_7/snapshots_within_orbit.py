import paths
import numpy as np
import yaml
import starry

from matplotlib import colors
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import (
    load_filter,
    planck,
    starry_intensity_to_bbtemp,
)

np.random.seed(42)

starry.config.lazy = False
starry.config.quiet = True


def initialize_featureless_map(T, wavelength_grid, ydeg=1):
    # Initialize star map
    map = starry.Map(ydeg=1, nw=len(wavelength_grid))
    Llam = (4 * np.pi) * np.pi * planck(T, wavelength_grid).value
    map.amp = Llam / 4
    return map


def get_lower_order_map(map, ydeg=2):
    assert map.ydeg > ydeg
    x = map._y * map.amp
    x = x[: (ydeg + 1) ** 2]
    map = starry.Map(ydeg, nw=map.nw)
    map[1:, :, :] = x[1:, :] / x[0]
    map.amp = x[0]

    return map


def initialize_map(ydeg, nw, x):
    map = starry.Map(ydeg, nw=nw)
    map[1:, :, :] = x[1:, :] / x[0]
    map.amp = x[0]
    return map


# System parameters
planet = "hd209"
filter_name = "f444w"

# Load orbital and system parameters
with open(
    paths.data/f"mapping_exo/system_parameters/{planet}/orbital_params_planet.yaml", "rb"
) as handle:
    params_p = yaml.safe_load(handle)
with open(
    paths.data/f"mapping_exo/system_parameters/{planet}/orbital_params_star.yaml", "rb"
) as handle:
    params_s = yaml.safe_load(handle)

# Load filter
filt = load_filter(paths.data/"mapping_exo/filter_files", name=f"{filter_name}")
mask = filt[1] > 0.002

# Wavelength grid for starry map (should match filter range)
wavelength_grid = np.linspace(4.5 - 1.2, 4.5 + 1.2, 50)

xs = np.load(paths.data/"output/mapping_exo/T341_1bar_ylms/coefficients_single_orbit.npy")
t_snapshots = np.load(paths.data/"output/mapping_exo/T341_1bar_ylms/times_single_orbit.npy")

porb = 3.5
t_snapshots = (t_snapshots - t_snapshots[0]) / porb

# Load simulation snapshots as starry maps
ydeg = 20

snapshots_maps = [initialize_map(ydeg, len(wavelength_grid), x) for x in xs]
snapshots_maps_quadrupole = [get_lower_order_map(map, ydeg=2) for map in snapshots_maps]

snapshots_maps_bbtemp = [
    starry_intensity_to_bbtemp(
        map.render(res=250, projection="Mollweide"), wavelength_grid
    )
    for map in snapshots_maps
]

snapshots_maps_quadrupole_bbtemp = [
    starry_intensity_to_bbtemp(
        map.render(res=250, projection="Mollweide"), wavelength_grid
    )
    for map in snapshots_maps_quadrupole
]

fig, ax = plt.subplots(2, 5, figsize=(17, 4), gridspec_kw={"wspace": 0.04})


map = starry.Map(ydeg)
norm = colors.Normalize(vmin=1500, vmax=1700)

for i, im in enumerate(snapshots_maps):
    map.show(
        image=snapshots_maps_bbtemp[i],
        projection="Mollweide",
        cmap="OrRd",
        norm=norm,
        ax=ax[0, i],
    )
    map.show(
        image=snapshots_maps_quadrupole_bbtemp[i],
        projection="Mollweide",
        cmap="OrRd",
        norm=norm,
        ax=ax[1, i],
    )


cax = ax[1, 2].inset_axes([-0.25, -0.2, 1.5, 0.1], transform=ax[1, 2].transAxes)

fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap="OrRd"),
    cax=cax,
    orientation="horizontal",
    label="blackbody temperature [K]",
    fraction=0.8,
)

for i, a in enumerate(ax[0]):
    a.set_title("$t/P = {:.2f}$".format(t_snapshots[i]))

fig.suptitle(
    "Hydro simulation snapshots at $l=25$ (top) and $l=2$ (bottom)", x=0.5, y=1.04
)

for a in ax.reshape(-1):
    a.set_rasterization_zorder(0)

fig.savefig(paths.figures/"snapshots_within_orbit.pdf", bbox_inches="tight")
