import paths
import numpy as np
from jax.config import config
import jax.numpy as jnp
from jax import vmap

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from caustics.utils import *
from caustics.extended_source_magnification import (
    _images_of_source_limb,
)
from caustics.integrate import _brightness_profile


config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

npts_limb = 80
u1 = 0.5
rho = 0.05
w_center = 1.5 * rho * jnp.exp(1j * jnp.pi / 4)

images, images_mask, images_parity = _images_of_source_limb(
    w_center,
    rho,
    npts=npts_limb,
    nlenses=1,
)
# Per image parity
parity = images_parity[:, 0]

# Set last point to be equal to first point
contours = jnp.hstack([images, images[:, 0][:, None]])
tail_idcs = jnp.array([images.shape[1] - 1, images.shape[1] - 1])

# Evaluate surface brightness on grid
xgrid_im, ygrid_im = jnp.meshgrid(
    np.linspace(-0.1, 1.3, 500), jnp.linspace(-0.1, 1.3, 500)
)
zgrid = xgrid_im + 1j * ygrid_im
I_eval = _brightness_profile(zgrid, rho, w_center, u1=u1, nlenses=1)

# Evaluate the P and Q integrals for a single image and point on contour
z0 = vmap(lambda idx, contour: contour.sum() / (idx + 1))(tail_idcs, contours)
idx_image = 0
idx_point = 70
z_origin = z0[idx_image]
z_limb = contours[idx_image, idx_point]

x_linspace = jnp.linspace(jnp.real(z_origin), jnp.real(z_limb), 200)
y_linspace = jnp.linspace(jnp.imag(z_origin), jnp.imag(z_limb), 200)

P_integrand = _brightness_profile(
    jnp.real(z_limb) + 1j * y_linspace, rho, w_center, u1=0.2, nlenses=1
)

Q_integrand = _brightness_profile(
    x_linspace + 1j * jnp.imag(z_limb), rho, w_center, u1=0.2, nlenses=1
)


axd = plt.figure(constrained_layout=True, figsize=(13, 8)).subplot_mosaic(
    """
    AB
    AC
    """,
    gridspec_kw={"wspace": 0.07, "hspace": 0.05, "width_ratios": [2, 1]},
)
im = axd["A"].pcolormesh(xgrid_im, ygrid_im, I_eval, cmap="Greys", zorder=-1)
# plt.colorbar(im, ax=ax)
axd["A"].set_aspect(1)
axd["A"].scatter(jnp.real(z_origin), jnp.imag(z_origin), marker="x", color="grey", s=150)


def plot_contour(ax, x, y,  color):
    s_min = 2.
    s_max = 8.
    s_vals = (np.linspace(s_min, s_max, len(x))**2)[::-1]
    ax.scatter(x, y, color=color, s=s_vals, marker='o', alpha=0.8, zorder=-1)
    ax.plot(x, y, color=color, linewidth=1., alpha=0.5, zorder=-1)

plot_contour(axd["A"], contours[0, :].real, contours[0, :].imag, 'C0')

# Plot P integral range and integrand
axd["A"].plot(
    jnp.real(z_limb) * jnp.ones_like(y_linspace),
    y_linspace,
    color="grey",
    linestyle="--",
    lw=2.5,
)
axd["B"].plot(y_linspace, P_integrand, color="k", lw=2.0)

# Plot Q integral range
axd["A"].plot(
    x_linspace,
    jnp.imag(z_limb) * jnp.ones_like(x_linspace),
    color="grey",
    linestyle="--",
    lw=2.5,
)
axd["C"].plot(x_linspace, Q_integrand, color="k", lw=2.0)
# Flip xaxis
axd["C"].invert_xaxis()
axd["A"].scatter(jnp.real(z_limb), jnp.imag(z_limb), marker="o", color="grey", s=15)

axd["A"].set(xlabel=r"$\mathrm{Re}(z)$", ylabel=r"$\mathrm{Im}(z)$")
axd["B"].set(xlabel=r"$\mathrm{Im}(z)$", ylabel=r"surface brightness")
axd["C"].set(xlabel=r"$\mathrm{Re}(z)$", ylabel=r"surface brightness")


axd["B"].set_title(r"$P$ integrand")
axd["C"].set_title(r"$Q$ integrand")
axd["A"].set_title("Image plane")

for _a in axd.values():
    _a.xaxis.set_minor_locator(AutoMinorLocator())
    _a.yaxis.set_minor_locator(AutoMinorLocator())
    _a.set_rasterization_zorder(0)

plt.savefig(paths.figures/"p_and_q_integrals.pdf", bbox_inches="tight")
