import numpy as np
from jax.config import config
import jax.numpy as jnp
from jax import random, jit, vmap, lax

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from caustics.utils import *
from caustics.extended_source_magnification import (
    _images_of_source_limb,
)
from caustics.integrate import _brightness_profile


config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

npts_limb = 200
niter_limb = 1
u1 = 0.3
rho = 0.1
w_center = 1.5 * rho * jnp.exp(1j * jnp.pi / 4)

images, images_mask, images_parity = _images_of_source_limb(
    w_center,
    rho,
    npts_init=npts_limb,
    niter=niter_limb,
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
idx_point = 160
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
axd["A"].scatter(jnp.real(z_origin), jnp.imag(z_origin), marker="x", color="C0", s=150)

# Plot P integral range and integrand
axd["A"].plot(
    jnp.real(z_limb) * jnp.ones_like(y_linspace),
    y_linspace,
    color="C0",
    linestyle="--",
    lw=2.5,
)
axd["B"].plot(y_linspace, P_integrand, color="k", lw=2.0)

# Plot Q integral range
axd["A"].plot(
    x_linspace,
    jnp.imag(z_limb) * jnp.ones_like(x_linspace),
    color="C0",
    linestyle="--",
    lw=2.5,
)
axd["C"].plot(x_linspace, Q_integrand, color="k", lw=2.0)
# Flip xaxis
axd["C"].invert_xaxis()
axd["A"].scatter(jnp.real(z_limb), jnp.imag(z_limb), marker="o", color="C0", s=60)

axd["A"].set(xlabel=r"$\mathrm{Re}(z)$", ylabel=r"$\mathrm{Im}(z)$")
axd["B"].set(xlabel=r"$\mathrm{Im}(z)$", ylabel=r"surface brightness")
axd["C"].set(xlabel=r"$\mathrm{Re}(z)$", ylabel=r"surface brightness")


axd["B"].set_title(r"$\mathcal{P}$ integrand")
axd["C"].set_title(r"$\mathcal{Q}$ integrand")
axd["A"].set_title("Image plane")
axd["A"].set_rasterization_zorder(0)

plt.savefig("p_and_q_integrals.pdf", bbox_inches="tight")
