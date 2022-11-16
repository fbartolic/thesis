import paths
import numpy as np
from jax.config import config
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from caustics import  critical_and_caustic_curves
from caustics.extended_source_magnification import _eval_images_sequentially

from caustics.utils import *

config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

def get_images(
    w0,
    rho,
    nlenses=2,
    npts=300,
    roots_itmax=2500,
    roots_compensated=False,
    **params,
):
    # Initial sampling on the source limb
    npts_init = int(0.5 * npts)
    theta = jnp.linspace(-np.pi, np.pi, npts_init - 1, endpoint=False)
    theta = jnp.pad(theta, (0, 1), constant_values=np.pi - 1e-8)
    z, z_mask, z_parity = _eval_images_sequentially(
        theta, w0, rho, nlenses, roots_compensated, roots_itmax, **params
    )

    return z, z_mask, z_parity


def plot_track(ax, x, y, mask, parity, color, connect=True, label=''):
    s_min = 3.
    s_max = 12
    s_vals = (np.linspace(s_min, s_max, len(x))**2)[::-1]

    for _x, _y, m, p, s in zip(x, y, mask, parity, s_vals):
        if m == True:
            marker ='o' 
        else:
            marker ='x'

        ax.scatter(_x, _y, s=s, alpha=0.6,color=color, marker=marker, zorder=-1)

    if connect:
        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.5, zorder=-1, label=label)




w0 = -0.67 + -0.027j
a, e1, e2, r3 = 0.698, 0.02809, 0.9687, -0.0197 - 0.95087j
rho = 5e-2

z, z_mask, z_parity = get_images(
    w0,
    rho,
    npts=200,
    nlenses=3,
    a=a,
    r3=r3,
    e1=e1,
    e2=e2,
)
cc, _ = critical_and_caustic_curves(
    nlenses=3, npts=1000, a=a, e1=e1, e2=e2, r3=r3, rho=rho
)

fig, ax = plt.subplot_mosaic(
    """
    ABCD
    EEEE
    """,
    figsize=(14, 12),
    gridspec_kw={'height_ratios': [1, 4], 'hspace':0.1, 'wspace':0.1},
)

ax['A'].set(xlim=(-0.0155, -0.011), ylim=(-0.961, -0.957))
ax['B'].set(xlim=(-0.0176, -0.01729), ylim=(-0.9493, -0.94895))
ax['C'].set(xlim=(0.733, 0.740), ylim=(-0.002, 0.005))
ax['D'].set(xlim=(0.694, 0.6988), ylim=(-0.0335, -0.0285))

labels = [
    r'$z^{(n)}_1$', r'$z^{(n)}_2$',  r'$z^{(n)}_3$', r'$z^{(n)}_4$', 
    r'$z^{(n)}_5$', r'$z^{(n)}_6$', r'$z^{(n)}_7$', r'$z^{(n)}_8$', 
    r'$z^{(n)}_9$', r'$z^{(n)}_{10}$']

for i in range(z.shape[0]):
    for _a in (ax.values()):
        plot_track(
            _a, z[i, :].real, z[i, :].imag, z_mask[i, :], z_parity[i, :], f'C{i}',
            label=labels[i]
            )
        _a.scatter(cc.real, cc.imag, c='black', marker='.', alpha=0.3, s=10, zorder=-1)
        _a.xaxis.set_minor_locator(AutoMinorLocator())
        _a.yaxis.set_minor_locator(AutoMinorLocator())
        _a.set_aspect(1)

ax['E'].set(xlim=(-1.8, 1.68), ylim=(-1.32, 1.05))
ax['E'].set(xlabel=r"$\mathrm{Re}(z)$", ylabel=r"$\mathrm{Im}(z)$")
ax['E'].legend(loc='lower right', fontsize=18)


for _a in (ax.values()):
    _a.set_rasterization_zorder(0)

# Save figure
fig.savefig(paths.figures/"initial_sampling.pdf", bbox_inches="tight")
