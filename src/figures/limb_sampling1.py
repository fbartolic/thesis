import numpy as np
from jax.config import config
import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


from caustics import images_point_source
from caustics.point_source_magnification import lens_eq_det_jac
from caustics.utils import *

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


def _images_of_source_limb(
    w_center,
    rho,
    nlenses=2,
    npts_init=500,
    niter=2,
    roots_itmax=2500,
    roots_compensated=False,
    **params,
):
    def images_and_mag(theta):
        z, z_mask = images_point_source(
            rho * jnp.exp(1j * theta) + w_center,
            nlenses=nlenses,
            roots_itmax=roots_itmax,
            roots_compensated=roots_compensated,
            **params,
        )
        det = lens_eq_det_jac(z, nlenses=nlenses, **params)
        parity = jnp.sign(det)
        mag = jnp.sum((1.0 / jnp.abs(det)) * z_mask, axis=0)
        return z, z_mask, mag, parity

    # Initial sampling on the source limb
    theta = jnp.linspace(-np.pi, np.pi, npts_init - 1, endpoint=False)
    theta = jnp.pad(theta, (0, 1), constant_values=np.pi - 1e-05)
    z, z_mask, mag, parity = images_and_mag(theta)

    # Refine sampling by placing geometrically fewer points each iteration
    # in the regions where the magnification gradient is largest
    npts_list = np.geomspace(2, npts_init, niter, endpoint=False, dtype=int)[::-1]
    key = random.PRNGKey(42)

    thetas = [theta]
    mags = [mag]

    for _npts in npts_list:
        # Resample theta
        delta_mag = jnp.gradient(mag)
        idcs_maxdelta = jnp.argsort(jnp.abs(delta_mag))[::-1][:_npts]
        theta_patch = 0.5 * (theta[idcs_maxdelta] + theta[idcs_maxdelta + 1])

        # Add small perturbation to avoid situations where there are duplicate values
        theta_patch += random.uniform(
            key, theta_patch.shape, maxval=1e-06
        )  # small perturbation

        z_patch, z_mask_patch, mag_patch, parity_patch = images_and_mag(theta_patch)

        thetas.append(theta_patch)
        mags.append(mag_patch)

        # Add to previous values and sort
        theta = jnp.concatenate([theta, theta_patch])
        sorted_idcs = jnp.argsort(theta)
        theta = theta[sorted_idcs]

        mag = jnp.concatenate([mag, mag_patch])[sorted_idcs]
        z = jnp.hstack([z, z_patch])[:, sorted_idcs]
        z_mask = jnp.hstack([z_mask, z_mask_patch])[:, sorted_idcs]
        parity = jnp.hstack([parity, parity_patch])[:, sorted_idcs]

    return z, z_mask, parity, thetas, mags


u1 = 0.3
rho = 0.3
w_center = 1.0 * rho * jnp.exp(1j * jnp.pi / 4)

npts_limb = 100
niter_limb = 3

images, images_mask, images_parity, thetas, mags = _images_of_source_limb(
    w_center,
    rho,
    npts_init=npts_limb,
    niter=niter_limb,
    nlenses=1,
)


fig, ax = plt.subplots(
    1, len(thetas), figsize=(16, 3), sharey=True, gridspec_kw={"wspace": 0.2}
)
for i in range(len(thetas)):
    ax[i].plot(jnp.rad2deg(thetas[i]), mags[i], f"C{i}o", alpha=0.7)
    ax[i].set_yscale("log")
    ax[i].set_ylim(1.0, 1.2e03)
    ax[i].set_xlim(-185.0, 185.0)
    ax[i].set(xlabel=r"$\theta_\mathrm{limb}$ [deg]")
    ax[i].xaxis.set_minor_locator(AutoMinorLocator())
    ax[i].set_xticks([-180, -90, 0, 90, 180])
    ax[i].set_title(f"iter = {i}")

ax[0].set_ylabel(ylabel=r"$\mathrm{magnification}$")

# Save figure
fig.savefig("limb_sampling1.pdf", bbox_inches="tight")
