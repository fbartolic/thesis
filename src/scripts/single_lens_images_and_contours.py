import paths
from functools import partial

import numpy as np
from jax.config import config
import jax.numpy as jnp
from jax import jit

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import colors

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from caustics import (
    lens_eq,
    mag_point_source,
)
from caustics.utils import *
from caustics.extended_source_magnification import (
    _images_of_source_limb,
)

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


@partial(jit, static_argnames=("nlenses"))
def _brightness_profile(z, rho, w_center, u1=0.0, nlenses=2, **params):
    w = lens_eq(z, nlenses=nlenses, **params)
    r = jnp.abs(w - w_center) / rho

    def safe_for_grad_sqrt(x):
        return jnp.sqrt(jnp.where(x > 0.0, x, 0.0))

    B_r = jnp.where(
        r <= 1.0,
        1 + safe_for_grad_sqrt(1 - r**2),
        0.01,
    )

    I = 3.0 / (3.0 - u1) * (u1 * B_r + 1.0 - 2.0 * u1)
    return I


def get_contours(w_center, rho, npts_limb=200):
    images, images_mask, images_parity = _images_of_source_limb(
        w_center,
        rho,
        npts_init=npts_limb,
        geom_factor=0.0,
        nlenses=1,
    )
    # Per image parity
    parity = images_parity[:, 0]

    # Set last point to be equal to first point
    contours = jnp.hstack([images, images[:, 0][:, None]])
    return contours, parity


npts_limb = 200
u1 = 0.3
rho = 0.3
w_center = 1.05 * rho * jnp.exp(1j * jnp.pi / 4)
l = 1.5

contours1, parity1 = get_contours(w_center, rho, npts_limb=npts_limb)

# Surface brightness in image plane
xgrid_im, ygrid_im = jnp.meshgrid(np.linspace(-l, l, 1000), jnp.linspace(-l, l, 1000))
zgrid = xgrid_im + 1j * ygrid_im
I_eval = _brightness_profile(zgrid, rho, w_center, u1=u1, nlenses=1)

# Source plane
xgrid, ygrid = jnp.meshgrid(
    np.linspace(-1.01 * rho, 2 * rho, 100), jnp.linspace(-1.01 * rho, 2 * rho, 100)
)
wgrid = xgrid + 1j * ygrid
mag_map = mag_point_source(wgrid, nlenses=1)


def plot_contour(ax, x, y, parity):
    if parity == 1.0:
        color = "C0"
    else:
        color = "C1"

    s_min = 1.0
    s_max = 12
    s_vals = (np.linspace(s_min, s_max, len(x)) ** 2)[::-1]
    ax.scatter(x, y, color=color, s=s_vals, marker=".", alpha=0.6, zorder=-1)
    ax.plot(x, y, color=color, linewidth=1.5, alpha=0.6)


fig, ax = plt.subplots(figsize=(8, 8))

ax.pcolormesh(xgrid_im, ygrid_im, I_eval, cmap="Greys", antialiased=True, zorder=-1)
ax.set_aspect(1)
ax.set(xlim=(-1.3, 1.8), ylim=(-1.6, 1.35))
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plot_contour(ax, jnp.real(contours1[0]), jnp.imag(contours1[0]), parity1[0])
plot_contour(ax, jnp.real(contours1[1]), jnp.imag(contours1[1]), parity1[1])


ax_in = inset_axes(
    ax,
    width="20%",  # width = 30% of parent_bbox
    height="20%",  # height : 1 inch
    #                    loc="lower right",
    bbox_to_anchor=(0.05, -0.55, 0.9, 0.9),
    bbox_transform=ax.transAxes,
    #                    borderpad=4.
)
ax_in.xaxis.set_minor_locator(AutoMinorLocator())
ax_in.yaxis.set_minor_locator(AutoMinorLocator())

ax_in.pcolormesh(xgrid, ygrid, mag_map, cmap="Greys", norm=colors.LogNorm(), zorder=-1)
ax_in.set_aspect(1)

circle = plt.Circle(
    (w_center.real, w_center.imag), rho, edgecolor="k", fill=False, alpha=0.7
)
ax_in.add_artist(circle)
ax_in.set_xlabel(r"$\mathrm{Re}(w)$")
ax_in.set_ylabel(r"$\mathrm{Im}(w)$")
ax_in.set_rasterization_zorder(0)

ax.set_rasterization_zorder(0)
ax_in.set_rasterization_zorder(0)

ax.set_ylabel(r"$\mathrm{Im}(z)$")
ax.set_xlabel(r"$\mathrm{Re}(z)$")
ax.set_title("Single lens images and contours")

fig.savefig(paths.figures/"single_lens_images_and_contours.pdf", bbox_inches="tight")
