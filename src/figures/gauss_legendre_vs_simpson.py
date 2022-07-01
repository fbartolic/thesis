from functools import partial

import numpy as np
from jax.config import config
import jax.numpy as jnp
from jax import random, jit, vmap, lax

import matplotlib.pyplot as plt

from caustics.utils import *
from caustics.extended_source_magnification import (
    _images_of_source_limb,
)
from caustics.integrate import _brightness_profile

from scipy.special import roots_legendre

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


@jit
def simpson_quadrature(x, y):
    """
    Compute an integral using Simpson's 1/3 rule assuming that x is uniformly
    distributed. `len(x)` has to be an odd number.
    """
    # len(x) must be odd
    h = (x[-1] - x[0]) / (len(x) - 1)
    return h / 3.0 * jnp.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)


@partial(jit, static_argnames=("npts"))
def compute_p_and_q(
    w_center,
    rho,
    contours,
    parity,
    tail_idcs,
    u1=0.0,
    npts=201,
):
    def P(_, y0, xl, yl):
        # Construct grid in z2 and evaluate the brightness profile at each point
        y = jnp.linspace(y0 * jnp.ones_like(xl), yl, npts)

        integrands = _brightness_profile(xl + 1j * y, rho, w_center, u1=u1, nlenses=1)
        I = simpson_quadrature(y, integrands)
        return -0.5 * I

    def Q(x0, _, xl, yl):
        # Construct grid in z1 and evaluate the brightness profile at each point
        x = jnp.linspace(x0 * jnp.ones_like(xl), xl, npts)

        integrands = _brightness_profile(x + 1j * yl, rho, w_center, u1=u1, nlenses=1)
        I = simpson_quadrature(x, integrands)
        return 0.5 * I

    # We choose the centroid of each contour to be lower limit for the P and Q
    # integrals
    z0 = vmap(lambda idx, contour: contour.sum() / (idx + 1))(tail_idcs, contours)
    x0, y0 = jnp.real(z0), jnp.imag(z0)

    # Select k and (k + 1)th elements
    contours_k = contours
    contours_kp1 = jnp.pad(contours[:, 1:], ((0, 0), (0, 1)))
    contours_k = vmap(lambda idx, contour: contour.at[idx].set(0.0))(
        tail_idcs, contours_k
    )

    # Compute the integral using the midpoint rule
    x_k = jnp.real(contours_k)
    y_k = jnp.imag(contours_k)

    x_kp1 = jnp.real(contours_kp1)
    y_kp1 = jnp.imag(contours_kp1)

    x_mid = 0.5 * (x_k + x_kp1)
    y_mid = 0.5 * (y_k + y_kp1)

    Pmid = vmap(P)(x0, y0, x_mid, y_mid)
    Qmid = vmap(Q)(x0, y0, x_mid, y_mid)

    return Pmid, Qmid


@partial(jit, static_argnames=("npts", "transform"))
def compute_p_and_q_gl(
    w_center,
    rho,
    contours,
    parity,
    tail_idcs,
    u1=0.0,
    npts=50,
):
    # Compute the Legendre roots and weights for use in Gaussian quadrature
    x_gl, w_gl = roots_legendre(npts)
    x_gl, w_gl = jnp.array(x_gl), jnp.array(w_gl)

    def P(_, y0, xl, yl):
        # Construct grid in z2 and evaluate the brightness profile at each point
        a = y0 * jnp.ones_like(xl)  # lower limits
        b = yl  # upper limits

        # Rescale domain for Gauss-Legendre quadrature
        A = 0.5 * (b - a)
        y_eval = 0.5 * (b - a) * x_gl[:, None] + 0.5 * (b + a)

        # Integrate
        f_eval = _brightness_profile(xl + 1j * y_eval, rho, w_center, u1=u1, nlenses=1)

        I = jnp.sum(A * w_gl[:, None] * f_eval, axis=0)

        return -0.5 * I

    def Q(x0, _, xl, yl):
        a = x0 * jnp.ones_like(xl)
        b = xl

        # Rescale domain for Gauss-Legendre quadrature
        A = 0.5 * (b - a)
        x_eval = 0.5 * (b - a) * x_gl[:, None] + 0.5 * (b + a)

        # Integrate
        f_eval = _brightness_profile(x_eval + 1j * yl, rho, w_center, u1=u1, nlenses=1)

        I = jnp.sum(A * w_gl[:, None] * f_eval, axis=0)

        return 0.5 * I

    # We choose the centroid of each contour to be lower limit for the P and Q
    # integrals
    z0 = vmap(lambda idx, contour: contour.sum() / (idx + 1))(tail_idcs, contours)
    x0, y0 = jnp.real(z0), jnp.imag(z0)

    # Select k and (k + 1)th elements
    contours_k = contours
    contours_kp1 = jnp.pad(contours[:, 1:], ((0, 0), (0, 1)))
    contours_k = vmap(lambda idx, contour: contour.at[idx].set(0.0))(
        tail_idcs, contours_k
    )

    # Compute the integral using the midpoint rule
    x_k = jnp.real(contours_k)
    y_k = jnp.imag(contours_k)

    x_kp1 = jnp.real(contours_kp1)
    y_kp1 = jnp.imag(contours_kp1)

    x_mid = 0.5 * (x_k + x_kp1)
    y_mid = 0.5 * (y_k + y_kp1)

    Pmid = vmap(P)(x0, y0, x_mid, y_mid)
    Qmid = vmap(Q)(x0, y0, x_mid, y_mid)

    return Pmid, Qmid


def compute_relative_error(rho, u1=0.2, npts=51, npts_limb=500, niter_limb=1):
    #    w_center = 1.7*rho*jnp.exp(1j*jnp.pi/4)
    w_center = 1.0 * rho * jnp.exp(1j * jnp.pi / 4)

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

    P_truth, Q_truth = compute_p_and_q(
        w_center, rho, contours, parity, tail_idcs, u1=u1, npts=1001
    )

    P1, Q1 = compute_p_and_q(
        w_center, rho, contours, parity, tail_idcs, u1=u1, npts=npts
    )
    P2, Q2 = compute_p_and_q_gl(
        w_center, rho, contours, parity, tail_idcs, u1=u1, npts=npts
    )

    err_P1 = jnp.abs((P1 - P_truth) / P_truth).mean(axis=0)
    err_P2 = jnp.abs((P2 - P_truth) / P_truth).mean(axis=0)

    return err_P1, err_P2


npts = 101
err_simps1, err_gl1 = compute_relative_error(
    1e-01, u1=0.2, npts=npts, npts_limb=300, niter_limb=5
)
err_simps2, err_gl2 = compute_relative_error(
    1e-02, u1=0.2, npts=npts, npts_limb=300, niter_limb=5
)
err_simps3, err_gl3 = compute_relative_error(
    1e-03, u1=0.2, npts=npts, npts_limb=300, niter_limb=5
)

fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True, gridspec_kw={"wspace": 0.1})

ax[0].plot(err_simps1, color="grey", lw=0.7, label="Simpson")
ax[0].plot(err_gl1, color="k", lw=1, alpha=0.8, label="Gauss-Legendre")


ax[1].plot(err_simps2, color="grey", lw=0.7, label="Simpson")
ax[1].plot(err_gl2, color="k", lw=1, alpha=0.8, label="Gauss-Legendre")


ax[2].plot(err_simps3, color="grey", lw=0.7, label="Simpson")
ax[2].plot(err_gl3, color="k", lw=1, alpha=0.8, label="Gauss-Legendre")


for a in ax:
    a.set_yscale("log")
    a.set(xlim=(0, 400), ylim=(5e-6, 5e-3))
    a.set(xlabel="countour point index")
    a.set_xticks(np.arange(0, 500, 100))

ax[0].set_title(r"$\rho_\star = 10^{-1}$")
ax[1].set_title(r"$\rho_\star = 10^{-2}$")
ax[2].set_title(r"$\rho_\star = 10^{-3}$")

ax[0].legend(prop={"size": 14})
ax[0].set(ylabel="relative error")

# Save as pdf
fig.savefig("gauss_legendre_vs_simpson.pdf", bbox_inches="tight")
