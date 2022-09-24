import paths

import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import AutoMinorLocator

from caustics import (
    critical_and_caustic_curves,
    images_point_source,
)

from caustics.utils import *
from caustics.multipole import mag_hexadecapole
from caustics.lightcurve import _caustics_proximity_test

import MulensModel as mm

def mag_vbb_binary(w0, rho, a, e1, u1=0., accuracy=5e-05):
    e2 = 1 - e1
    x_cm = (e1 - e2)*a
    bl = mm.BinaryLens(e2, e1, 2*a)
    return bl.vbbl_magnification(w0.real - x_cm, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1)

# System parameters
rho = 5e-03
a = 0.8
e1 = 0.95
e2 = 1. - e1

critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, a=a, e1=e1, npts=100)


_x = jnp.linspace(-0.5, -0.5 + 1.5, int(3.7*200))
_y = jnp.linspace(-0.2, -0.2 + 0.4, 200)
xgrid, ygrid = jnp.meshgrid(_x, _y)
wgrid = xgrid + 1j*ygrid

# Full calculation with VBB
mags_ref = np.zeros_like(wgrid).astype(float)

for i in range(wgrid.shape[0]):
    for j in range(wgrid.shape[1]):
        w_center = wgrid[i, j]
        mags_ref[i, j]  = mag_vbb_binary(w_center, rho, a, e1, u1=0.0)

# Evaluate hex approx. and the test
z, z_mask = images_point_source(wgrid, nlenses=2, a=a, e1=e1)
mu_multi, delta_mu_multi = mag_hexadecapole(z, z_mask, rho, nlenses=2, a=a,e1=e1)
err_hex = jnp.abs(mu_multi - mags_ref)/mags_ref

test = _caustics_proximity_test(
    wgrid, z, z_mask, rho, delta_mu_multi, 
    nlenses=2,  a=a, e1=e1
)

fig, ax = plt.subplots(figsize=(14, 10))

cmap1 = colors.ListedColormap(['grey', 'white'])
cmap2 = colors.ListedColormap(['white', 'red'])

im = ax.pcolormesh(xgrid, ygrid, test, cmap=cmap1, alpha=0.9, zorder=-1)
im = ax.pcolormesh(xgrid, ygrid, err_hex > 1e-04, cmap=cmap2, alpha=0.5, zorder=-1)

for cc in caustic_curves:
    ax.plot(cc.real, cc.imag, color="black", lw=0.7)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='grey', label=r'Tests evaluate to "False"', alpha=0.8),
    Patch(facecolor='red', label=r'$\epsilon_\mathrm{rel}>10^{-4}$', alpha=0.6)
]

ax.legend(handles=legend_elements, fontsize=14)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_aspect(1)
ax.set(xlim=(-0.5, 1.), ylim=(-0.2, 0.2))
ax.set(xlabel=r"$\mathrm{Re}(w)$", ylabel=r"$\mathrm{Im}(w)$")
ax.set_rasterization_zorder(0)
fig.savefig(paths.figures/"extended_source_test.pdf", bbox_inches="tight")
