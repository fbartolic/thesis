import paths
import numpy as np

import os
from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator

from jax import random
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp

from caustics import (
    mag,
    mag_point_source,
    critical_and_caustic_curves,
)
from caustics.linalg import weighted_least_squares

import numpyro
import numpyro.distributions as dist
from numpyro_ext.optim import optimize as optimize_numpyro_ext

def magnitudes_to_fluxes(mag, mag_err, zeropoint=22):
    flux = 10. ** (0.4 * (zeropoint - mag))
    err_flux = mag_err * flux * np.log(10.) * 0.4
    return flux, err_flux

def get_data(path):
    t = Table.read(os.path.join(path, "phot.dat"), format="ascii")

    # Remove additional columns
    t.columns[0].name = "HJD"
    t.columns[1].name = "mag"
    t.columns[2].name = "mag_err"
    t.keep_columns(("HJD", "mag", "mag_err"))
    return t

lc = get_data(paths.data/"microlensing/blg-0039/")

# Trim
lc['HJD'] = lc['HJD'] - 2450000
lc = lc[lc['HJD'] > 7400]
fobs, ferr = magnitudes_to_fluxes(lc["mag"], lc["mag_err"])
t = lc['HJD']
t, fobs, ferr = jnp.array(t), jnp.array(fobs), jnp.array(ferr)
t0_est = 7832

def model(t, t0_est, fobs, ferr):
    ln_s = numpyro.sample("ln_s", dist.Uniform(jnp.log(1e-02), jnp.log(3.)))
    ln_q = numpyro.sample("ln_q", dist.Uniform(jnp.log(1e-05), jnp.log(1.)))
    ln_t0 = numpyro.sample("ln_t0", dist.Uniform(jnp.log(t0_est - 50), jnp.log(t0_est + 50)))
    ln_u0 = numpyro.sample("ln_u0", dist.Uniform(jnp.log(1e-03), 2.))
    ln_tE = numpyro.sample("ln_tE", dist.Uniform(jnp.log(10.), jnp.log(400.)))
    alpha = numpyro.sample("alpha", dist.Uniform(-np.pi, np.pi))
    ln_rho = numpyro.sample("ln_rho", dist.Uniform(jnp.log(5e-04), jnp.log(0.5)))
    s, q, t0, u0, tE, rho = jnp.exp(ln_s), jnp.exp(ln_q), jnp.exp(ln_t0),jnp.exp(ln_u0), jnp.exp(ln_tE), jnp.exp(ln_rho)

    w_x = (t - t0)/tE*jnp.cos(alpha) - u0*jnp.sin(alpha)
    w_y = (t - t0)/tE*jnp.sin(alpha) + u0*jnp.cos(alpha)
    w_points = w_x + 1j*w_y

    A = mag(w_points, rho, nlenses=2, s=s, q=q)

    M = jnp.stack([A - 1., jnp.ones_like(A)]).T
    beta = weighted_least_squares(fobs, ferr, M)
    fpred = (M @ beta).reshape(-1)

    numpyro.sample("obs", dist.Normal(fpred, ferr), obs=fobs)


def fit_map(key, t0_est, params_init):
    opt = optimize_numpyro_ext(model, start=params_init)
    return opt(key, t=t, t0_est=t0_est, fobs=fobs, ferr=ferr)


key = random.PRNGKey(0)
key, subkey = random.split(key)

start1 = {
    'ln_t0':8.96613361,
    'ln_tE':5.09365964,
    'ln_u0':-4.61386331,
    'ln_s':0.49577905,
    'ln_q':-1.59125415,
    'alpha':-1.34008613,
    'ln_rho':-6.46841773,
}

start2 = {
    'ln_s':jnp.log(0.779892),
    'ln_q':jnp.log(0.118918),
    'ln_t0':jnp.log(7830.33),
    'ln_u0':jnp.log(0.131204),
    'ln_tE':jnp.log(180.226),
    'alpha':1.72459 - np.pi,
    'ln_rho':jnp.log(0.00124506),
}

start3 = {
    'ln_s':jnp.log(2.21836),
    'ln_q':jnp.log(0.339234),
    'ln_t0':jnp.log(7847.51),
    'ln_u0':jnp.log(0.375093),
    'ln_tE':jnp.log(241.946),
    'alpha':4.56247 - np.pi,
    'ln_rho':jnp.log(0.000985705),
}


start4 = {
    'ln_s':jnp.log(0.532289),
    'ln_q':jnp.log(0.328923),
    'ln_t0':jnp.log(7832.83),
    'ln_u0':jnp.log(0.0497645),
    'ln_tE':jnp.log(195.881),
    'alpha':1.73302 - np.pi,
    'ln_rho':jnp.log(0.00114756),
}

map_params1 = fit_map(subkey, t0_est, start1)
map_params2 = fit_map(subkey, t0_est, start2)
map_params3 = fit_map(subkey, t0_est, start3)
map_params4 = fit_map(subkey, t0_est, start4)

map_params1.pop('obs', None)
map_params2.pop('obs', None)
map_params3.pop('obs', None)
map_params4.pop('obs', None)


f_ll = lambda p: numpyro.infer.util.log_likelihood(
    model,
    p, 
    t=t, t0_est=t0_est, fobs=fobs,ferr=ferr, 
)['obs'].sum()
chi_sq_vals = -2*np.array([f_ll(map_params1), f_ll(map_params2), f_ll(map_params3), f_ll(map_params4)])

def eval_fpred(t, fobs, ferr, params):
    ln_s = params['ln_s']
    ln_q = params['ln_q']
    ln_t0 = params['ln_t0']
    ln_u0 = params['ln_u0']
    ln_tE = params['ln_tE']
    alpha = params['alpha']
    ln_rho = params['ln_rho']
    s, q, t0, u0, tE, rho = jnp.exp(ln_s), jnp.exp(ln_q), jnp.exp(ln_t0),jnp.exp(ln_u0), jnp.exp(ln_tE), jnp.exp(ln_rho)

    # Evaluate model at observed times
    w_x = (t - t0)/tE*jnp.cos(alpha) - u0*jnp.sin(alpha)
    w_y = (t - t0)/tE*jnp.sin(alpha) + u0*jnp.cos(alpha)
    w_points = w_x + 1j*w_y 
    A = mag(w_points, rho, nlenses=2, s=s, q=q)
    M = jnp.stack([A - 1., jnp.ones_like(A)]).T
    beta = weighted_least_squares(fobs, ferr, M)
    fpred = (M @ beta).reshape(-1)

    # Evaluate model on finer grid
    _t = jnp.linspace(t[0], t[-1], 5000)
    w_x = (_t - t0)/tE*jnp.cos(alpha) - u0*jnp.sin(alpha)
    w_y = (_t - t0)/tE*jnp.sin(alpha) + u0*jnp.cos(alpha)
    w_points_dense = w_x + 1j*w_y 
    A = mag(w_points_dense, rho, nlenses=2, s=s, q=q)
    M = jnp.stack([A - 1., jnp.ones_like(A)]).T
    fpred_dense = (M @ beta).reshape(-1)

    critical_curves, caustic_curves = critical_and_caustic_curves(nlenses=2, s=s, q=q, npts=100)

    return _t, fobs - fpred, fpred_dense, w_points, caustic_curves

t_dense, res1, fpred_dense1, w_points1, caustic_curves1 = eval_fpred(t, fobs, ferr, map_params1)
_, res2, fpred_dense2, w_points2, caustic_curves2 = eval_fpred(t, fobs, ferr, map_params2)
_, res3, fpred_dense3, w_points3, caustic_curves3 = eval_fpred(t, fobs, ferr, map_params3)
_, res4, fpred_dense4, w_points4, caustic_curves4 = eval_fpred(t, fobs, ferr, map_params4)

fig, ax = plt.subplots(8, 1, figsize=(14, 10), sharex=True,
    gridspec_kw={'height_ratios': [2, 1, 2, 1, 2, 1, 2, 1]})

ax_lc = ax[[0, 2, 4, 6]]
ax_res = ax[[1, 3, 5, 7]]

ax_lc[0].plot(t_dense, fpred_dense1, color='C0', label='Mode 1')
ax_res[0].errorbar(t, res1, ferr,  marker='o', color='k', linestyle='None', alpha=0.2) 

ax_lc[1].plot(t_dense, fpred_dense2, color='C1', label='Mode 2')
ax_res[1].errorbar(t, res2, ferr,  marker='o', color='k', linestyle='None', alpha=0.2)

ax_lc[2].plot(t_dense, fpred_dense3, color='C2', label='Mode 3')
ax_res[2].errorbar(t, res3, ferr,  marker='o', color='k', linestyle='None', alpha=0.2)

ax_lc[3].plot(t_dense, fpred_dense4, color='C3', label='Mode 4')
ax_res[3].errorbar(t, res4, ferr,  marker='o', color='k', linestyle='None', alpha=0.2)

for a in ax_lc:
    a.errorbar(t, fobs, ferr,  marker='o', color='k', linestyle='None', alpha=0.2) 
    a.set_xlim(7788, 7900)
    a.grid(0.5)
    a.set_ylabel('Flux')
    a.legend(fontsize=16, handlelength=0, handletextpad=0)

for a in ax_res:
    a.set_ylim(-6, 6)
    a.grid(alpha=0.5)
    a.set_ylabel("Residuals")

for a in ax:
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())
    
ax[-1].set_xlabel('Time [HJD - 2450000 days]')
fig.savefig(paths.figures/'binary_lens_modes_fluxes.pdf', bbox_inches='tight')


# Plot (s, q) plane
fig, ax = plt.subplots(2, 2, figsize=(9, 10), sharey=True, gridspec_kw={'hspace':0.3})
ax = ax.reshape(-1)

for cc in caustic_curves1:
    ax[0].plot(cc.real, cc.imag, color='k', lw=0.5)
for cc in caustic_curves2:
    ax[1].plot(cc.real, cc.imag, color='k', lw=0.5)
for cc in caustic_curves3:
    ax[2].plot(cc.real, cc.imag, color='k', lw=0.5)
for cc in caustic_curves4:
    ax[3].plot(cc.real, cc.imag, color='k', lw=0.5)


xgrid, ygrid = jnp.meshgrid(
    jnp.linspace(-1.3, 1.7, 300),
    jnp.linspace(-1.5, 1.5, 300)
)
wgrid = xgrid + 1j*ygrid

mag_ps1 = mag_point_source(wgrid, nlenses=2, s=jnp.exp(map_params1['ln_s']), q=jnp.exp(map_params1['ln_q']))
mag_ps2 = mag_point_source(wgrid, nlenses=2, s=jnp.exp(map_params2['ln_s']), q=jnp.exp(map_params2['ln_q']))
mag_ps3 = mag_point_source(wgrid, nlenses=2, s=jnp.exp(map_params3['ln_s']), q=jnp.exp(map_params3['ln_q']))
mag_ps4 = mag_point_source(wgrid, nlenses=2, s=jnp.exp(map_params4['ln_s']), q=jnp.exp(map_params4['ln_q']))

for a in ax:
    a.set_aspect(1)
    a.set(xlim=(-1.3, 1.7), ylim=(-1.5, 1.5))


def plot_circles(ax, rho, w_points, color):
    circles = [
        plt.Circle((xi,yi), radius=rho,
        fill=False, facecolor=None, color=color, zorder=-1) for xi,yi in zip(w_points.real, w_points.imag)
    ]
    c = mpl.collections.PatchCollection(circles, match_original=True, alpha=0.4, zorder=-1)
    ax.add_collection(c)


ax[0].pcolormesh(xgrid, ygrid, mag_ps1, cmap='Greys', norm=mpl.colors.LogNorm(vmax=50), zorder=-1)
ax[1].pcolormesh(xgrid, ygrid, mag_ps2, cmap='Greys', norm=mpl.colors.LogNorm(vmax=50), zorder=-1)
ax[2].pcolormesh(xgrid, ygrid, mag_ps3, cmap='Greys', norm=mpl.colors.LogNorm(vmax=50), zorder=-1)
ax[3].pcolormesh(xgrid, ygrid, mag_ps4, cmap='Greys', norm=mpl.colors.LogNorm(vmax=50), zorder=-1)

plot_circles(ax[0], np.exp(map_params1['ln_rho']), w_points1, 'C0')
plot_circles(ax[1], np.exp(map_params2['ln_rho']), w_points2, 'C1')
plot_circles(ax[2], np.exp(map_params3['ln_rho']), w_points3, 'C2')
plot_circles(ax[3], np.exp(map_params4['ln_rho']), w_points4, 'C3')

ax[0].set_title("Mode 1, $\chi^2$ = {:.1f}".format(chi_sq_vals[0]))
ax[1].set_title("Mode 2, $\chi^2$ = {:.1f}".format(chi_sq_vals[1]))
ax[2].set_title("Mode 3, $\chi^2$ = {:.1f}".format(chi_sq_vals[2]))
ax[3].set_title("Mode 4, $\chi^2$ = {:.1f}".format(chi_sq_vals[3]))


ax[0].set_ylabel(r"$\mathrm{Im}(w)$")
ax[2].set_ylabel(r"$\mathrm{Im}(w)$")


for a in ax:
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())
    a.set_xlabel(r"$\mathrm{Re}(w)$")
    a.set_rasterization_zorder(0)


fig.savefig(paths.figures/'binary_lens_modes.pdf', bbox_inches='tight')
