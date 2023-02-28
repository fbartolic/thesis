import paths
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import random, lax


import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from caustics import (
    critical_and_caustic_curves,
    mag_extended_source,
)

from caustics.utils import *
from caustics.integrate import *

import MulensModel as mm

def mag_vbb_binary(w0, rho, s, q, u1=0.0, accuracy=1e-05):
    e1 = 1/(1 + q)
    e2 = 1 - e1
    bl = mm.BinaryLens(e1, e2, s)
    return bl.vbbl_magnification(
        w0.real, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1
    )

@partial(jit, static_argnames=("npts_limb", "npts_ld", "limb_darkening"))
def mag_binary(w_points, rho, s, q, u1=0., npts_limb=300, npts_ld=100, limb_darkening=False):
    def body_fn(_, w):
        mag = mag_extended_source(
            w,
            rho,
            nlenses=2,
            npts_limb=npts_limb,
            limb_darkening=limb_darkening,
            npts_ld=npts_ld,
            u1=u1,
            s=s,
            q=q,
        )
        return 0, mag
    _, mags = lax.scan(body_fn, 0, w_points)
    return mags



# Parameters
s, q = 0.9, 0.2
u1 = 0.7

# 1000  points on caustic curve
npts = 20
critical_curves, caustic_curves = critical_and_caustic_curves(
    npts=npts, nlenses=2, s=s, q=q
)
caustic_curves = caustic_curves.reshape(-1)

# Accuracy of integration
acc_vbb = 1e-05
npts_limb = 500
npts_ld = 150


# Compute relative error w.r.t. VBB for different values of rho 
mags_vbb_list = []
mags_list = []

rho_list = [1.,1e-01, 1e-02, 1e-03]

for rho in rho_list:
    print(f"rho = {rho}")

    # Generate 1000 random test points within 2 source radii away from the caustic points 
    key = random.PRNGKey(5)
    key, subkey1, subkey2 = random.split(key, num=3)
    phi = random.uniform(subkey1, caustic_curves.shape, minval=-np.pi, maxval=np.pi)
    r = random.uniform(subkey2, caustic_curves.shape, minval=0., maxval=2*rho)
    w_test = caustic_curves + r*np.exp(1j*phi)

    mags_vbb = np.array(
        [
            mag_vbb_binary(complex(w), rho, s, q, u1=u1, accuracy=acc_vbb)
            for w in w_test
        ]
    )
    mags = mag_binary(w_test, rho, s, q, u1=u1, npts_limb=npts_limb, npts_ld=npts_ld, limb_darkening=True)
    mags_vbb_list.append(mags_vbb)
    mags_list.append(mags)

# Make plot
fig, ax = plt.subplots(1,len(rho_list), figsize=(14, 4), sharey=True,
    gridspec_kw={'wspace':0.3})


labels = [
    r"$\rho_\star=10^{0}$", r"$\rho_\star=10^{-1}$", r"$\rho_\star=10^{-2}$", r"$\rho_\star=10^{-3}$", r"$\rho_\star=10^{-4}$"
]
for i in range(len(rho_list)):
    mags = mags_list[i]
    mags_vbb = mags_vbb_list[i]
    ax[i].plot(jnp.abs((mags - mags_vbb)/mags_vbb), 'k-', alpha=1., zorder=-1, lw=0.8)
    ax[i].xaxis.set_minor_locator(AutoMinorLocator())
    ax[i].yaxis.set_minor_locator(AutoMinorLocator())
    ax[i].set_yscale('log')
    ax[i].set_title(labels[i])
    ax[i].set_ylim(5e-06, 2e-03)
    ax[i].set_rasterization_zorder(0)

ax[0].set_ylabel("Relative error")
fig.text(0.512, 0.0, r"Point index", ha='center')
fig.savefig(paths.figures/"mag_limbdark_accuracy.pdf", bbox_inches="tight")