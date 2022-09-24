import paths
import numpy as np
from jax.config import config
import jax.numpy as jnp
from jax import random

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import colors

from caustics import images_point_source, critical_and_caustic_curves, mag_point_source
from caustics.point_source_magnification import lens_eq_det_jac
from caustics.extended_source_magnification import _images_point_source_sequential 

from caustics.utils import *


config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

def _images_of_source_limb(
    w0,
    rho,
    nlenses=2,
    npts=300,
    niter=10,
    roots_itmax=2500,
    roots_compensated=False,
    **params,
):
    key = random.PRNGKey(0)
    key1, key2 = random.split(key)

    def fn(theta, z_init):
        # Add a small perturbation to z_init to avoid situations where
        # there is convergence to the exact same roots when difference in theta
        # is very small and z_init is very close to the exact roots
        u1 = random.uniform(key1, shape=z_init.shape, minval=-1e-6, maxval=1e-6)
        u2 = random.uniform(key2, shape=z_init.shape, minval=-1e-6, maxval=1e-6)
        z_init = z_init + u1 + u2 * 1j

        z, z_mask = images_point_source(
            rho * jnp.exp(1j * theta) + w0,
            nlenses=nlenses,
            roots_itmax=2500,
            roots_compensated=roots_compensated,
            z_init=z_init.T,
            custom_init=True,
            **params,
        )
        det = lens_eq_det_jac(z, nlenses=nlenses, **params)
        z_parity = jnp.sign(det)
        mag = jnp.sum((1.0 / jnp.abs(det)) * z_mask, axis=0)
        return z, z_mask, z_parity, mag

    # Initial sampling on the source limb
    npts_init = int(0.5 * npts)
    theta = jnp.linspace(-np.pi, np.pi, npts_init - 1, endpoint=False)
    theta = jnp.pad(theta, (0, 1), constant_values=np.pi - 1e-8)
    z, z_mask = _images_point_source_sequential(
        rho * jnp.exp(1j * theta) + w0, nlenses=nlenses, roots_itmax=roots_itmax, **params
    )
    det = lens_eq_det_jac(z, nlenses=nlenses, **params)
    z_parity = jnp.sign(det)
    mag = jnp.sum((1.0 / jnp.abs(det)) * z_mask, axis=0)

    theta_init, z_init, z_mask_init, mag_init = theta, z, z_mask, mag

    # Refine sampling by adding npts_init additional points a fraction
    # 1 / niter at a time
    npts_additional = int(0.5 * npts)


    delta_z = jnp.abs(z[:, 1:] - z[:, :-1])
    delta_z_mask = jnp.logical_or(z_mask[:, 1:], z_mask[:, :-1])
    delta_z = jnp.where(
        delta_z_mask,
        delta_z,
        jnp.zeros_like(delta_z.real),
    )

    n = int(npts_additional / niter)

    theta_list, z_list, z_mask_list, mag_list = [], [], [], []

    for i in range(niter):
        # Find the indices (along the contour axis) with the biggest distance
        # gap for consecutive points under the condition that at least
        # one of the two points is a real image
        delta_z = jnp.abs(z[:, 1:] - z[:, :-1])
        delta_z = jnp.where(
            (~z_mask[:, 1:]) & (~z_mask[:, :-1]),
            jnp.zeros_like(delta_z.real),
            delta_z,
        )
        idcs_max = jnp.argsort(jnp.max(delta_z, axis=0))[::-1][:n]

        # Add new points at the midpoints of the top-ranking intervals
        theta_new = 0.5 * (theta[idcs_max] + theta[idcs_max + 1])
        z_new, z_mask_new, z_parity_new, mag_new = fn(theta_new, z[:, idcs_max])

        delta_z = delta_z.at[:, idcs_max].set(jnp.abs(z_new - z[:, idcs_max]))
        delta_z = jnp.insert(delta_z, idcs_max + 1, jnp.abs(z[:, idcs_max + 1] -z_new), axis=1)

        theta = jnp.insert(theta, idcs_max + 1, theta_new, axis=0)
        z = jnp.insert(z, idcs_max + 1, z_new, axis=1)
        z_mask = jnp.insert(z_mask, idcs_max + 1, z_mask_new, axis=1)
        z_parity = jnp.insert(z_parity, idcs_max + 1, z_parity_new, axis=1)

        # Recompute delta_z_mask because we've updated delta_z
        delta_z_mask = jnp.logical_or(z_mask[:, 1:], z_mask[:, :-1])
        delta_z = jnp.where(
            delta_z_mask,
            delta_z,
            jnp.zeros_like(delta_z.real),
        )

        theta_list.append(theta_new)
        z_list.append(z_new)
        z_mask_list.append(z_mask_new)
        mag_list.append(mag_new)

    # Get rid of duplicate values of images which may occur in rare cases
    # by adding a very small random perturbation to the images
    _, c = jnp.unique(z.reshape(-1), return_counts=True, size=len(z.reshape(-1)))
    c = c.reshape(z.shape)
    mask_dup = c > 1
    z = jnp.where(
        mask_dup,
        z + random.uniform(key, shape=z.shape, minval=1e-9, maxval=1e-9),
        z,
    )

    return (theta_init, z_init, z_mask_init, mag_init, 
    jnp.concatenate(theta_list), jnp.concatenate(z_list), jnp.concatenate(z_mask_list), jnp.concatenate(mag_list))

w0 = -0.67 + -0.027j
a, e1, e2, r3 = 0.698, 0.02809, 0.9687, -0.0197 - 0.95087j
rho = 5e-2

theta_init, z_init, z_mask_init, mag_init, theta_new, z_new, z_mask_new, mag_new = _images_of_source_limb(
    w0,
    rho,
    npts=200,
    niter=10,
    nlenses=3,
    a=a,
    r3=r3,
    e1=e1,
    e2=e2,
)
cc, _ = critical_and_caustic_curves(
    nlenses=3, npts=300, a=a, e1=e1, e2=e2, r3=r3, rho=rho
)

fig, ax = plt.subplot_mosaic(
    """
    AB
    CC
    """,
    figsize=(12, 13.5),
    gridspec_kw={'height_ratios': [1, 3], 'hspace':0.3},
)
# Source plane
xgrid, ygrid = jnp.meshgrid(
   w0.real +   np.linspace(-3*rho, 3*rho, 500), 
    w0.imag + jnp.linspace(-1.7*rho, 1.7*rho, 300))
wgrid = xgrid + 1j*ygrid
mag_map = mag_point_source(wgrid, nlenses=3, a=a, e1=e1, e2=e2, r3=r3)
ax['A'].pcolormesh(xgrid, ygrid, mag_map, cmap='Greys', zorder=-1, norm=colors.LogNorm())
circle = plt.Circle((w0.real, w0.imag), rho, edgecolor="k", fill=False, alpha=0.7, zorder=-1)
ax['A'].add_artist(circle)
ax['A'].set_aspect(1)

# Magnification
ax['B'].plot(
    jnp.rad2deg(theta_init), mag_init, marker='o', ls='', c='k', alpha=0.8, label='Initial',
    zorder=-1,
)
ax['B'].plot(
    jnp.rad2deg(theta_new), mag_new, marker='o', ls='', c='C1', alpha=0.5, label='Additional',
    zorder=-1,
)
ax['B'].set_xlim(-180, 180)
ax['B'].set_xticks([-180, -90, 0, 90, 180])
ax['B'].xaxis.set_minor_locator(AutoMinorLocator())
ax['B'].set_yscale('log')
ax['B'].set_ylim(2., 3e3)
#ax['B'].legend(fontsize=14, loc='upper left')
 
# Inset axes for images
ax_in1 = inset_axes(ax['C'],
    width="25%", # width = 30% of parent_bbox
    height="25%", # height : 1 inch
    loc="lower right",
    bbox_transform=ax['C'].transAxes,
    borderpad=2.
)
ax_in2 = inset_axes(ax['C'],
    width="25%", # width = 30% of parent_bbox
    height="25%", # height : 1 inch
    loc="upper right",
    bbox_transform=ax['C'].transAxes,
    borderpad=2.
)

for _a in ax['A'], ax['C'], ax_in1, ax_in2:
    _a.xaxis.set_minor_locator(AutoMinorLocator())
    _a.yaxis.set_minor_locator(AutoMinorLocator())
    _a.set_aspect(1)


# Initial images
mask = z_mask_init.reshape(-1)
x  = z_init.reshape(-1)[mask].real
y = z_init.reshape(-1)[mask].imag 

ax['C'].scatter(x, y, marker='o', alpha=0.8, c='k', zorder=-1, label="Initial")
ax_in1.scatter(x, y, marker='o', alpha=0.8, c='k', zorder=-1)
ax_in2.scatter(x, y, marker='o', alpha=0.8, c='k', zorder=-1)

# Additional images
mask = z_mask_new.reshape(-1)
x  = z_new.reshape(-1)[mask].real
y = z_new.reshape(-1)[mask].imag 

ax['C'].scatter(x, y, marker='o', alpha=0.5, c='C1', zorder=-1, label="Additional")
ax_in1.scatter(x, y, marker='o', alpha=0.5, c='C1', zorder=-1)
ax_in2.scatter(x, y, marker='o', alpha=0.5, c='C1', zorder=-1)


ax_in1.set(xlim=(-0.0152, -0.011), ylim=(-0.961, -0.957))
ax_in2.set(xlim=(0.733, 0.74), ylim=(-0.002, 0.005))

ax['C'].set(xlim=(-1.8, 1.4), ylim=(-1.2, 1.1))
ax['B'].set(xlabel=r'$\phi$ [deg]', ylabel=r'$\mathrm{magnification}$')

#ax['C'].legend(fontsize=14, loc='upper left')
ax['A'].set_title("Source plane")
ax['B'].set_title("Magnification along the limb")
ax['C'].set_title("Image plane")
ax['A'].set(xlabel=r"$\mathrm{Re}(w)$", ylabel=r"$\mathrm{Im}(w)$")
ax['C'].set(xlabel=r"$\mathrm{Re}(z)$", ylabel=r"$\mathrm{Im}(z)$")

# Critical curves
for z in cc:
    ax['C'].plot(z.real, z.imag, color='k', lw=0.7, zorder=-2)


for _a in ([ax['A'], ax['B'], ax['C'], ax_in1, ax_in2]):
    _a.set_rasterization_zorder(0)


# Save figure
fig.savefig(paths.figures/"adaptive_sampling.pdf", bbox_inches="tight")
