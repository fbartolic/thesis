import numpy as np
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

from caustics import critical_and_caustic_curves_binary

s_array = jnp.array([0.6, 1 / np.sqrt(2), 1.2, 2.0, 3.2])

fig, ax = plt.subplots(5, 2, figsize=(10, 18), sharex=True)
fig.subplots_adjust(hspace=0.15, wspace=0.3)

for i, s_ in enumerate(s_array):
    critical_curves, caustic_curves = critical_and_caustic_curves_binary(
        0.5 * s_, 0.5, npts=1000
    )

    ax[i, 0].scatter(
        critical_curves.real, critical_curves.imag, s=0.8, c="k", alpha=0.8
    )
    ax[i, 1].scatter(caustic_curves.real, caustic_curves.imag, s=0.8, c="k", alpha=0.8)


for a in ax.ravel():
    #    a.grid(True)
    a.set_xlim(-2.5, 2.5)
    a.set_ylim(-1.6, 1.6)
    a.set_aspect(1)

for a in ax[:, 0]:
    a.set_ylabel(r"$\mathrm{Im}(z)$")

for a in ax[:, 1]:
    a.set_ylabel(r"$\mathrm{Im}(w)$")

ax[-1, 0].set_xlabel(r"$\mathrm{Re}(z)$")
ax[-1, 0].set_xticks([-2, -1, 0, 1, 2])
ax[-1, 1].set_xlabel(r"$\mathrm{Re}(w)$")
ax[-1, 1].set_xticks([-2, -1, 0, 1, 2])

ax[0, 0].set_title(r"close", x=1.15, pad=10)
ax[0, 0].text(
    0.17,
    0.1,
    "$s=0.6$",
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax[0, 0].transAxes,
    fontsize=16,
)
ax[1, 0].set_title(r"close$\;\rightarrow\;$intermediate", x=1.15, pad=10)
ax[1, 0].text(
    0.17,
    0.1,
    "$s=1/\sqrt{2}$",
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax[1, 0].transAxes,
    fontsize=16,
)

ax[2, 0].set_title(r"intermediate", x=1.15, pad=10)
ax[2, 0].text(
    0.17,
    0.1,
    "$s=1.2$",
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax[2, 0].transAxes,
    fontsize=16,
)


ax[3, 0].set_title(r"intermediate$\;\rightarrow\;$wide", x=1.15, pad=10)
ax[3, 0].text(
    0.17,
    0.1,
    "$s=2.0$",
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax[3, 0].transAxes,
    fontsize=16,
)


ax[4, 0].set_title(r"wide", x=1.15, pad=10)
ax[4, 0].text(
    0.17,
    0.1,
    "$s=3.2$",
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax[4, 0].transAxes,
    fontsize=16,
)

for _a in ax[:, 1]:
    _a.set_yticklabels([])


fig.savefig("binary_lens_topology.pdf", bbox_inches="tight")
