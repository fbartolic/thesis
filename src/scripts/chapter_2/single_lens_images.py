import paths
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

import matplotlib.pyplot as plt
import matplotlib.colors as colors


def mag(z):
    u = jnp.abs(z) ** 2
    return (u ** 2 + 2) / (u * jnp.sqrt(u ** 2 + 4))


@jit
def brightness_profile(z, rho, w_center, u=0.1):
    w = z - 1 / jnp.conjugate(z)
    r = jnp.abs(w - w_center) / rho

    B_r = jnp.where(r <= 1.0, jnp.sqrt(1 - r ** 2), 0.0,)
    I = 3.0 / (3.0 - u) * (u * B_r + 1.0 - u)
    return I


# Source plane
xgrid, ygrid = jnp.meshgrid(np.linspace(-1, 1, 200), jnp.linspace(-1, 1, 200))
wgrid = xgrid + 1j * ygrid

mag = vmap(vmap(mag))(wgrid)

# Lens plane
rho = 0.1
w_center = 0.2

l = 1.5
xgrid_im, ygrid_im = jnp.meshgrid(np.linspace(-l, l, 500), jnp.linspace(-l, l, 500))
zgrid = xgrid_im + 1j * ygrid_im
I_eval = brightness_profile(zgrid, rho, w_center, u=0.2)

fig, ax = plt.subplots(2, 3, figsize=(15, 10), gridspec_kw={"hspace": 0.4})

for a in ax[0]:
    a.pcolormesh(
        xgrid, ygrid, mag, cmap="Greys", norm=colors.LogNorm(vmax=1e04), zorder=-1
    )

rho = 0.15
w1, w2, w3 = -0.4, -0.2, 0.0

I_eval = brightness_profile(zgrid, rho, w1, u=0.2)
ax[1, 0].pcolormesh(xgrid_im, ygrid_im, I_eval, cmap="Greys", zorder=-1)

I_eval = brightness_profile(zgrid, rho, w2, u=0.2)
ax[1, 1].pcolormesh(xgrid_im, ygrid_im, I_eval, cmap="Greys", zorder=-1)

I_eval = brightness_profile(zgrid, rho, w3, u=0.2)
ax[1, 2].pcolormesh(xgrid_im, ygrid_im, I_eval, cmap="Greys", zorder=-1)

circle = plt.Circle((w1, 0), rho, facecolor="k", alpha=0.4)
ax[0, 0].add_artist(circle)

circle = plt.Circle((w2, 0), rho, facecolor="k", alpha=0.4)
ax[0, 1].add_artist(circle)

circle = plt.Circle((w3, 0), rho, facecolor="k", alpha=0.4)
ax[0, 2].add_artist(circle)

ax[0, 1].set_title("Source plane", fontsize=20, pad=20)
ax[1, 1].set_title("Image plane", fontsize=20, pad=20)


for a in ax.reshape(-1):
    a.set_aspect(1)
    a.set_rasterization_zorder(0)

# Save to disk
fig.savefig(paths.figures/"single_lens_images.pdf", bbox_inches="tight", dpi=200)
