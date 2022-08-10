import paths
from functools import partial

from jax.config import config
import jax.numpy as jnp
from jax import jit, lax

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from caustics import mag_extended_source, images_point_source
from caustics.multipole import mag_hexadecapole


config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")


@partial(
    jit,
    static_argnames=("nlenses", "npts_limb", "niter_limb", "limb_darkening", "npts_ld"),
)
def mag_extended_source_vectorized(
    w_points,
    rho,
    limb_darkening=False,
    u1=0.0,
    nlenses=2,
    npts_limb=200,
    niter_limb=4,
    npts_ld=50,
):
    # Iterate over w_points and execute either the hexadecapole  approximation
    # or the full extended source calculation. `vmap` cannot be used here because
    # `lax.cond` executes both branches within vmap.
    mag_full = lambda w: mag_extended_source(
        w,
        rho,
        limb_darkening=limb_darkening,
        u1=u1,
        nlenses=nlenses,
        npts_limb=npts_limb,
        niter_limb=niter_limb,
        npts_ld=npts_ld,
    )

    def body_fn(_, x):
        w = x
        m = mag_full(w)
        return 0, m

    return lax.scan(body_fn, 0, w_points)[1]


# Parameters
u1 = 0.3
rho = 0.05

npts_limb = 150
niter_limb = 15
npts_ld = 100

w_points = jnp.linspace(0.0, 3.0 * rho, 200)

# Hex approx.
z, mask_z = images_point_source(w_points, nlenses=1)
mu_ps, mu_quad, mu_hex = mag_hexadecapole(z, mask_z, rho, u1=u1, nlenses=1)
mag_multi = mu_ps + mu_quad + mu_hex

# Full computation
mag_full = mag_extended_source_vectorized(
    w_points,
    rho,
    limb_darkening=True,
    u1=u1,
    nlenses=1,
    npts_limb=npts_limb,
    niter_limb=niter_limb,
    npts_ld=npts_ld,
)

fig, ax = plt.subplots(
    2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
ax[0].plot(
    jnp.real(w_points) / rho, mag_full, color="k", lw=2.0, label="full computation"
)
ax[0].plot(
    jnp.real(w_points) / rho,
    mag_multi,
    color="Grey",
    lw=1.5,
    label="hexadecapole\napproximation",
)
ax[1].plot(
    jnp.real(w_points) / rho, (mag_multi - mag_full) / mag_full, color="k", lw=2.0
)
ax[0].legend(prop={"size": 14})

ax[0].set_ylabel(r"$\mathrm{Magnification}$")
ax[1].set(xlabel=r"$\mathrm{Re(w)\,/\,\rho_\star}$", ylabel="Residual")
ax[0].set_ylim(5.0, 45.0)
ax[1].set_ylim(-1e-02, 1e-02)

for a in ax:
    a.axvline(2.0, color="k", ls="--", alpha=0.9, lw=0.7)
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())

fig.savefig(paths.figures/"single_lens_hex.pdf", bbox_inches="tight")
