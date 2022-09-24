import paths
import numpy as np
import jax.numpy as jnp
from jax.config import config

config.update("jax_enable_x64", True)


import matplotlib.pyplot as plt
import matplotlib.colors as colors

from caustics import mag_point_source

# Lens postions
a = 0.4
r3 = 0.3 - 0.8j

# Lens mass ratios
e1 = 0.9
e2 = 0.05
e3 = 1.0 - e1 - e2

psi = jnp.arctan2(r3.imag, r3.real)
s = 0.5*a
q = e2/e1
q3 = (1 - e1-e2)/e1


# Define 2D grid in the source plane
npts = 1000

x = jnp.linspace(-0.1, 1, npts)
y = jnp.linspace(-0.45, 0.5, npts)
xgrid, ygrid = jnp.meshgrid(x, y)
wgrid = xgrid + 1j * ygrid

mag = mag_point_source(wgrid, s=s, q=q, q3=q3, r3=jnp.abs(r3), psi=psi, nlenses=3)

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.pcolormesh(
    jnp.real(wgrid),
    jnp.imag(wgrid),
    mag,
    cmap="Greys",
    norm=colors.LogNorm(vmax=150),
    zorder=-1,
)

ax.set_aspect(1)
ax.set(xlabel="$\mathrm{Re}(w)$", ylabel="$\mathrm{Im}(w)$")
plt.colorbar(im, label="magnification")
ax.set_rasterization_zorder(0)

# Save to disc
fig.savefig(paths.figures/"triple_lens_caustics.pdf", bbox_inches="tight", dpi=200)

