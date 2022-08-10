import paths
import numpy as np
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)


import matplotlib.pyplot as plt
import matplotlib.colors as colors

from caustics import mag_point_source_binary

q = 5e-03
e1 = q / (1 + q)

s_array = jnp.array([0.7, 1 / 0.7])

t = jnp.linspace(-10.0, 10.0, 600)
tE = 30


mosaic = """
    AC
    BC
    """
fig = plt.figure(figsize=(14, 6), constrained_layout=True)
ax_dict = fig.subplot_mosaic(mosaic, gridspec_kw={"wspace": 0.05})

ax_maps = [ax_dict["A"], ax_dict["B"]]

xgrid, ygrid = jnp.meshgrid(jnp.linspace(-1.2, 0.2, 1000), jnp.linspace(-0.3, 0.3, 300))
wgrid = xgrid + 1j * ygrid

for i, s_ in enumerate(s_array):
    a = 0.5 * s_
    x_cm = (2 * e1 - 1) * a
    mag = mag_point_source_binary(wgrid, a, e1)
    im = ax_maps[i].pcolormesh(
        xgrid,
        ygrid,
        mag,
        cmap="Greys",
        norm=colors.LogNorm(vmin=1, vmax=200),
        zorder=-1,
    )


# Trajectory s
t0 = -1.5
u0 = -0.36
alpha = jnp.deg2rad(10.0)

x_traj = jnp.cos(alpha) * u0 - jnp.sin(alpha) * (t - t0) / tE
y_traj = jnp.sin(alpha) * u0 + jnp.cos(alpha) * (t - t0) / tE
w_traj = x_traj + 1j * y_traj
mag_traj = mag_point_source_binary(w_traj, 0.5 * s_array[0], e1)
ax_maps[0].plot(x_traj, y_traj, color="k", alpha=0.5, linestyle="dashed")
ax_dict["C"].plot(t, mag_traj, color=f"k", lw=2.5, label="s = 0.7")

# Trajectory 1/s
t0 = -3.7
u0 = -0.7167
alpha = jnp.deg2rad(10.8)

x_traj = jnp.cos(alpha) * u0 - jnp.sin(alpha) * (t - t0) / tE
y_traj = jnp.sin(alpha) * u0 + jnp.cos(alpha) * (t - t0) / tE
w_traj = x_traj + 1j * y_traj
mag_traj = mag_point_source_binary(w_traj, 0.5 * s_array[1], e1)
ax_maps[1].plot(x_traj, y_traj, color="k", alpha=0.5, linestyle="dashed")
ax_dict["C"].plot(t, mag_traj, "C1--", lw=2.5, label="s = 1/0.7")


for _a, txt in zip(ax_maps, ["0.7", "1/0.7"]):
    _a.text(
        0.98,
        0.12,
        f"$s = {txt}$",
        horizontalalignment="right",
        verticalalignment="center",
        transform=_a.transAxes,
        fontsize=16,
    )


for a in ax_maps:
    a.set_aspect(1)
    a.set(xlim=(xgrid[0, 0], xgrid[0, -1]), ylim=(ygrid[0, 0], ygrid[-1, 0]))

    # Formatting each yaxis
    minorLocator = plt.MultipleLocator(0.1)
    minorFormatter = plt.FormatStrFormatter("%1.1f")
    a.xaxis.set_minor_locator(minorLocator)
    a.set_rasterization_zorder(0)

ax_maps[0].set_xticklabels([])
ax_dict["C"].set_aspect(0.18)
ax_dict["C"].set(xlim=(-6, 6), xlabel="time", ylabel="magnification")
ax_dict["C"].legend(loc="upper right", fontsize=14)
ax_maps[1].set_xlabel(r"$\mathrm{Re}(w)$")
ax_maps[0].set_ylabel(r"$\mathrm{Im}(w)$")
ax_maps[1].set_ylabel(r"$\mathrm{Im}(w)$")


# Save to file
fig.savefig(paths.figures/"close_wide_degeneracy.pdf", bbox_inches="tight")
