
import paths
from functools import partial
import numpy as np
from jax.config import config
import jax.numpy as jnp
from jax import  jit, vmap

import matplotlib.pyplot as plt

from caustics import lens_eq
from caustics.utils import *
from caustics.extended_source_magnification import (
    _images_of_source_limb,
)
from caustics.integrate import (
    _brightness_profile, _integrate_gauss_legendre
)
import scipy

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

@jit
def simpson_quadrature(x, y):
    """
    Compute an integral using Simpson's 1/3 rule assuming that x is uniformly
    distributed. `len(x)` has to be an odd number.
    """
    # len(x) must be odd
    h = (x[-1] - x[0]) / (len(x) - 1)
    return h / 3.0 * jnp.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)


def compute_P_integrand_scipy(
    w_center, rho, u1=0.0, nlenses=1,  npts_limb=100, 
):
    z, z_mask, z_parity = _images_of_source_limb(
        w_center,
        rho,
        npts=npts_limb,
        nlenses=1,
    )

    # Set last point to be equal to first point
    z= jnp.hstack([z, z[:, 0][:, None]])
    tail_idcs = jnp.array([z.shape[1] - 1, z.shape[1] - 1])

    # Compute the Legendre roots and weights for use in Gaussian quadrature
    def P(_, y0, xl, yl):
        # Construct grid in z2 and evaluate the brightness profile at each point
        a, b = y0 * jnp.ones_like(xl), yl  # lower and upper limits

        # Integrate
        I = []
        for i in range(len(xl)):
            f = lambda y: np.array(_brightness_profile(
                xl[i] + 1j * y, rho, w_center, u1=u1, nlenses=nlenses, 
            ))
            val, err = scipy.integrate.quad(f, a[i], b[i], limit=1000, epsrel=1e-12)
            I.append(val)
        return -0.5 * jnp.array(I)

    # We choose the centroid of each contour to be lower limit for the P and Q
    # integrals
    z0 = vmap(lambda idx, contour: contour.sum() / (idx + 1))(tail_idcs, z)
    # Evaluate the P and Q integrals using Trapezoidal rule
    _P = jnp.stack([P(_z10, _z20, _z1, _z2) for _z10, _z20, _z1, _z2 in zip(z0.real, z0.imag, z.real, z.imag)])
    return _P


@partial(jit, static_argnames=("npts", "rule", "npts_limb", "niter_limb"))
def compute_P_integrand(
    w_center, rho, u1=0.0, npts=50, rule="legendre", npts_limb=100, niter_limb=15
):
    z, z_mask, z_parity = _images_of_source_limb(
        w_center,
        rho,
        npts=npts_limb,
        nlenses=1,
    )
    # Set last point to be equal to first point
    z = jnp.hstack([z, z[:, 0][:, None]])
    tail_idcs = jnp.array([z.shape[1] - 1, z.shape[1] - 1])

    def P(_, y0, xl, yl):
        # Construct grid in z2 and evaluate the brightness profile at each point
        a = y0*jnp.ones_like(xl) # lower limits
        b = yl # upper limits

        if rule == "simpson":
            y = jnp.linspace(y0 * jnp.ones_like(xl), yl, npts)

            integrands = _brightness_profile(xl + 1j * y, rho, w_center, u1=u1, nlenses=1)
            I = simpson_quadrature(y, integrands)

        elif rule =="split_legendre":
            abs_delta = jnp.abs(b - a)
            y_split = jnp.where(
                b > a,
                b - 2*rho,
                b + 2*rho,
            )
            y_split = jnp.where(
                (0.5*abs_delta <= 2*rho) & (b > a),
                a + 0.5*abs_delta,
                y_split
            )
            y_split = jnp.where(
                (0.5*abs_delta <= 2*rho) & ~(b>a),
                a - 0.5*abs_delta,
                y_split
            )

            # Integrate from a to a + 0.99|b - a|
            f = lambda y: _brightness_profile(
                xl + 1j * y, rho, w_center, u1=u1, nlenses=1 
            )
            npts1 = int(npts/2)
            npts2 = npts - npts1
            I1 = _integrate_gauss_legendre(f, a, y_split, n=npts1)

            # Integrate from a + 0.99|b - a| to b
            I2 = _integrate_gauss_legendre(f, y_split, b, n=npts2)
            I = I1 + I2

        else:
            f = lambda y: _brightness_profile(
                xl + 1j * y, rho, w_center, u1=u1, nlenses=1 
            )
            I = _integrate_gauss_legendre(f, a, b, n=npts)

        return -0.5 * I

    # We choose the centroid of each contour to be lower limit for the P and Q
    # integrals
    z0 = vmap(lambda idx, z: z.sum() / (idx + 1))(tail_idcs, z)

    # Evaluate the P and Q integrals using Trapezoidal rule
    _P = vmap(P)(z0.real, z0.imag, z.real, z.imag)

    return _P


npts_limb = 100 
niter_limb = 1
u1 = 0.8
rho_list = [1e-01, 1e-02, 1e-03]
P_true_list = []

w_center = 0.5 + 0j

for rho in rho_list:
    print("rho = ", rho)
    P = compute_P_integrand_scipy(w_center*rho, rho, npts_limb=npts_limb,  u1=u1)
    P_true_list.append(P)

npts = 101

P_simps_list = []
P_legendre_list = []
P_split_legendre_list = []

for i, rho in enumerate(rho_list):
    print("rho = ", rho)

    P_simps = compute_P_integrand(
        w_center*rho, rho, rule="simpson", npts_limb=npts_limb, niter_limb=niter_limb, u1=u1, npts=npts
    )
    P_legendre = compute_P_integrand(
        w_center*rho, rho,  npts_limb=npts_limb, niter_limb=niter_limb, u1=u1, npts=npts
    )
    P_legendre_split = compute_P_integrand(
        w_center*rho, rho, rule="split_legendre", npts_limb=npts_limb, niter_limb=niter_limb, u1=u1, npts=npts
    )

    P_simps_list.append(P_simps)
    P_legendre_list.append(P_legendre)
    P_split_legendre_list.append(P_legendre_split)

fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for i in range(3):
    err_simps = jnp.abs(P_simps_list[i] - P_true_list[i])/jnp.abs(P_true_list[i])
    err_legendre = jnp.abs(P_legendre_list[i] - P_true_list[i])/jnp.abs(P_true_list[i])
    err_split_legendre = jnp.abs(P_split_legendre_list[i] - P_true_list[i])/jnp.abs(P_true_list[i])

    ax[i].plot(
        jnp.max(err_simps, axis=0), label="Simpson", c='k', alpha=0.8)
    ax[i].plot(
        jnp.max(err_legendre, axis=0), label="Gauss-Legendre", ls='--', c='k', alpha=0.5)
    ax[i].plot(
        jnp.max(err_split_legendre, axis=0), label="Gauss-Legendre (split)",  ls='-', lw=0.6, c='grey')

for a in ax:
    a.set_yscale("log")
    a.set_ylim(1e-08, 1e-01)
#    a.set(xlim=(0, 400), ylim=(5e-6, 5e-3))
    a.set(xlabel="countour point index")
#    a.set_xticks(np.arange(0, 500, 100))

ax[0].set_title(r"$\rho_\star = 10^{-1}$")
ax[1].set_title(r"$\rho_\star = 10^{-2}$")
ax[2].set_title(r"$\rho_\star = 10^{-3}$")

ax[-1].legend(prop={"size": 14})
ax[0].set(ylabel="relative error")


# Save figure
fig.savefig(paths.figures/"quadrature_comparison.pdf", bbox_inches="tight")
