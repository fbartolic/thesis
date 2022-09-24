import paths
import numpy as np
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)


import matplotlib.pyplot as plt
import matplotlib.colors as colors

from caustics import mag_point_source

q = 5e-03
e1 = q / (1 + q)
s_array = [0.8, 0.9, 1.0, 1.2, 1.4]

fig, ax = plt.subplots(len(s_array), figsize=(12, 14), sharex=True)
fig.subplots_adjust(hspace=0.05)

for i, s_ in enumerate(s_array):

    xgrid, ygrid = jnp.meshgrid(
        jnp.linspace(-1, 1.0, 1500), jnp.linspace(-0.2, 0.2, 300)
    )
    wgrid = xgrid + 1j * ygrid
    a = 0.5 * s_
    x_cm = (2 * e1 - 1) * a
    mag = mag_point_source(wgrid, a=a, e1=e1, nlenses=2)
    im = ax[i].pcolormesh(
        xgrid,
        ygrid,
        mag,
        cmap="Greys",
        norm=colors.LogNorm(vmin=1, vmax=200),
        zorder=-1,
    )


for i, s_ in enumerate(s_array):
    ax[i].scatter(0.5 * s_, [0.0], color="C0", marker="o", label=r"planet", alpha=0.7)
    ax[i].scatter(
        -0.5 * s_, [0.0], color="C1", marker="*", label=r"star", s=100, alpha=0.7
    )
    ax[i].text(
        0.98,
        0.12,
        f"$s={s_}$",
        horizontalalignment="right",
        verticalalignment="center",
        transform=ax[i].transAxes,
        fontsize=16,
    )


for a in ax.ravel():
    a.set_aspect(1)

    # Formatting each yaxis
    minorLocator = plt.MultipleLocator(0.1)
    minorFormatter = plt.FormatStrFormatter("%1.1f")
    a.xaxis.set_minor_locator(minorLocator)
    a.set_rasterization_zorder(0)


ax[0].legend(prop={"size": 12})
ax[2].set_ylabel(r"$\mathrm{Im}(w)$")
ax[-1].set_xlabel(r"$\mathrm{Re}(w)$")

# Add colorbar on the right side
plt.colorbar(
    im,
    cax=fig.add_axes([0.93, 0.25, 0.03, 0.5]),
    orientation="vertical",
    label="magnification",
)

# Save to file
fig.savefig(paths.figures/"planetary_caustics.pdf", bbox_inches="tight")
