import paths
import numpy as np
from jax.config import config
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from caustics import critical_and_caustic_curves
from caustics.extended_source_magnification import (
    _images_of_source_limb,
    _get_segments,
    _contours_from_closed_segments,
    _contours_from_open_segments,
)
from caustics.utils import *

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

def plot_contour(ax, x, y,  color):
    s_min = 1.
    s_max = 12
    s_vals = (np.linspace(s_min, s_max, len(x))**2)[::-1]
    ax.scatter(x, y, color=color, s=s_vals, marker='o', alpha=0.8, zorder=-1)
    ax.plot(x, y, color=color, linewidth=1., alpha=0.5, zorder=-1)

w0 = -0.67 + -0.027j
a, e1, e2, r3 = 0.698, 0.02809, 0.9687, -0.0197 - 0.95087j
rho = 5e-2

z, z_mask, z_parity = _images_of_source_limb(
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

segments_closed, segments_open, _ = _get_segments(z, z_mask, z_parity, nlenses=3)
cc, _ = critical_and_caustic_curves(
    nlenses=3, npts=300, a=a, e1=e1, e2=e2, r3=r3, rho=rho
)
contours1, contours1_p = _contours_from_closed_segments(segments_closed)
contours2, contours2_p = _contours_from_open_segments(segments_open)

segments_list = []
for i in range(len(segments_open)):
    if not jnp.all(segments_open[i] == 0 + 0j):
        segments_list.append(segments_open[i])
for i in range(len(segments_closed)):
    if not jnp.all(segments_closed[i] == 0 + 0j):
        segments_list.append(segments_closed[i])

fig, ax = plt.subplots(2,1, figsize=(12, 18))

# TOP PANEL
# Inset axes for images
ax_in1 = inset_axes(ax[0],
    width="25%", # width = 30% of parent_bbox
    height="25%", # height : 1 inch
    loc="lower right",
    bbox_transform=ax[0].transAxes,
    borderpad=2.
)
ax_in2 = inset_axes(ax[0],
    width="25%", # width = 30% of parent_bbox
    height="25%", # height : 1 inch
    loc="upper right",
    bbox_transform=ax[0].transAxes,
    borderpad=2.
)

for i in range(len(segments_list)):
    segment = segments_list[i]
    for _a in (ax[0], ax_in1, ax_in2):
        tidx = last_nonzero(jnp.abs(segment[0, :]))
        plot_contour(
            _a,
            segment[0, 1:].real[:tidx],
            segment[0, 1:].imag[:tidx],
            f'C{i%10}',
        )

for z in cc:
    ax[0].plot(z.real, z.imag, color='k', lw=0.7) 

ax[0].set_title("Contour segments")

# BOTTOM PANEL 
# Inset axes for images
ax_in3 = inset_axes(ax[1],
    width="25%", # width = 30% of parent_bbox
    height="25%", # height : 1 inch
    loc="lower right",
    bbox_transform=ax[1].transAxes,
    borderpad=2.
)
ax_in4 = inset_axes(ax[1],
    width="25%", # width = 30% of parent_bbox
    height="25%", # height : 1 inch
    loc="upper right",
    bbox_transform=ax[1].transAxes,
    borderpad=2.
)

colors = {True: 'C1', False: 'C0'}

idx = 3
tidx = last_nonzero(jnp.abs(contours1[idx]))
_p = bool(contours1_p[idx] > 0)
plot_contour(
    ax_in4,
    contours1[idx, 1:].real[:tidx],
    contours1[idx, 1:].imag[:tidx],
    colors[_p],
)

idx = 8
_p = bool(contours1_p[idx] > 0)
plot_contour(
    ax_in3,
    contours1[idx, 1:].real[:tidx],
    contours1[idx, 1:].imag[:tidx],
    colors[_p],
)

for idx in range(contours2.shape[0]):
    _p = bool(contours2_p[idx] > 0)
    tidx = last_nonzero(jnp.abs(contours2[idx]))
    plot_contour(
        ax[1],
        contours2[idx, 1:].real[:tidx],
        contours2[idx, 1:].imag[:tidx],
        colors[_p],
    )


for _a in (ax_in1, ax_in3):
    _a.set(xlim=(-0.0152, -0.011), ylim=(-0.961, -0.957))
for _a in (ax_in2, ax_in4):
    _a.set(xlim=(0.733, 0.74), ylim=(-0.002, 0.005))
for _a in (ax_in1, ax_in2, ax_in3, ax_in4):
    _a.set_aspect(1)
    _a.xaxis.set_minor_locator(AutoMinorLocator())
    _a.yaxis.set_minor_locator(AutoMinorLocator())


for _a in ax:
    _a.set(xlim=(-1.8, 1.5), ylim=(-1.2, 1.1))
    _a.set_aspect(1)
    _a.xaxis.set_minor_locator(AutoMinorLocator())
    _a.yaxis.set_minor_locator(AutoMinorLocator())
    _a.set(xlabel=r"$\mathrm{Re}(z)$", ylabel=r"$\mathrm{Im}(z)$")

ax[1].set_title("Closed contours")

for _a in (ax[0], ax[1], ax_in1, ax_in2):
    _a.set_rasterization_zorder(0)

# Save figure
fig.savefig(paths.figures/"segments_and_contours.pdf", bbox_inches="tight")
