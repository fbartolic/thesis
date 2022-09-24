import paths

from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import jit, jacfwd 

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator

from caustics import (
    mag,
    critical_and_caustic_curves,
)

# Parameters
a = 0.45
e1 = 0.8
e2 = 1. - e1
q = e1/e2
rho = 1e-02
alpha = jnp.deg2rad(45.)
u0 = 0.15 # COM frame
t0 = 50.
tE = 23.

t = jnp.linspace(30, 70, 2000) # days

@jit
def get_mag(params):
    s, q, rho, alpha, u0, t0, tE = params
    a = 0.5*s
    e1 = q/(1 + q)

    x_cm = (e1 - e2)*a # b
    u_mp = u0 - x_cm*jnp.sin(alpha)
    t_mp = t0 - tE*(u0/jnp.tan(alpha) - u_mp/jnp.tan(alpha))

    w_x = (t - t_mp)/tE*jnp.cos(alpha) - u_mp*jnp.sin(alpha)
    w_y = (t - t_mp)/tE*jnp.sin(alpha) + u_mp*jnp.cos(alpha)
    w_points = w_x + 1j*w_y

    return w_points, mag(w_points, rho, nlenses=2, a=a, e1=e1)

s = 2*a
params = jnp.array([s, q, rho, alpha, u0, t0, tE])
w_points, A = get_mag(params)

# Evaluate the Jacobian at every point
mag_jac = jit(jacfwd(lambda params: get_mag(params)[1]))
jac_eval = mag_jac(params)

fig, ax = plt.subplots(
    8, 1,
    figsize=(14, 14),
    gridspec_kw={'height_ratios': [4, 1, 1, 1, 1, 1,1, 1], 'wspace':0.3},
    sharex=True,
)
# Inset axes for images
ax_in = inset_axes(ax[0],
    width="60%", # 
    height="60%", 
    bbox_transform=ax[0].transAxes,
    bbox_to_anchor=(-0.4, 0.05, .9, .9),
)
ax_in.set_aspect(1)
ax_in.set(xlabel="$\mathrm{Re}(w)$", ylabel="$\mathrm{Im}(w)$")

critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, a=a, e1=e1, npts=100)
for cc in caustic_curves:
    ax_in.plot(cc.real, cc.imag, color='black', lw=0.7)

circles = [
    plt.Circle((xi,yi), radius=rho, fill=False, facecolor=None, zorder=-1) for xi,yi in zip(w_points.real, w_points.imag)
]
c = mpl.collections.PatchCollection(circles, match_original=True, alpha=0.05)

ax_in.add_collection(c)
ax_in.set_aspect(1)
ax_in.set(xlim=(-1., 1.2), ylim=(-0.8, 1.))

ax[0].plot(t, A, color='black', lw=2)

for i, _a in enumerate(ax[1:]):
    _a.plot(t, jac_eval[:, i], lw=2., color='C0')

labels = [
    r'$A(t)$',
    r'$\frac{\partial A}{\partial s}$', r'$\frac{\partial A}{\partial q}$', r'$\frac{\partial A}{\partial \rho}$',
     r'$\frac{\partial A}{\partial \alpha}$', r'$\frac{\partial A}{\partial u_0}$', r'$\frac{\partial A}{\partial t_0}$', 
     r'$\frac{\partial A}{\partial t_E}$'
]

labelx = -0.07  # axes coords
for i, _a in enumerate(ax):
    _a.set_ylabel(
        labels[i], 
        rotation=0, 
        verticalalignment='center',
        horizontalalignment='right',
        fontsize=20,
    )
    _a.yaxis.set_label_coords(labelx, 0.5)
    _a.xaxis.set_minor_locator(AutoMinorLocator())
    _a.yaxis.set_minor_locator(AutoMinorLocator())
    _a.set(xlim=(30, 70))

ax[1].set_ylim(-550, 550)
ax[2].set_ylim(-55, 55)
ax[3].set_ylim(-1200, 1200)
ax[4].set_ylim(-550, 550)
ax[5].set_ylim(-1300, 1300)
ax[6].set_ylim(-55, 55)
ax[7].set_ylim(-12, 12)

ax[-1].set_xlabel('$t$ [days]')
ax_in.set_rasterization_zorder(0)
fig.savefig(paths.figures/"lightcurve_gradients.pdf", bbox_inches="tight")
