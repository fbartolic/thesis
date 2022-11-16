import paths
import numpy as np

import os
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from caustics.linalg import weighted_least_squares

import VBBinaryLensing
VBBL = VBBinaryLensing.VBBinaryLensing()

def mag_vbb_binary(w0, rho, s, q, u1=0.0, accuracy=1e-05):
    return VBBL.BinaryMag2(s, q, w0.real, w0.imag, rho)

def ll_fn_vbb(params, pointwise=False):
    ln_s, ln_q, ln_t0, ln_u0, ln_tE, alpha, ln_rho =(
        params['ln_s'], params['ln_q'], params['ln_t0'], params['ln_u0'], params['ln_tE'], params['alpha'], params['ln_rho']
    )
    s, q, t0, u0, tE, rho = np.exp(ln_s), np.exp(ln_q), np.exp(ln_t0),np.exp(ln_u0), np.exp(ln_tE), np.exp(ln_rho)

    w_x = (t - t0)/tE*np.cos(alpha) - u0*np.sin(alpha)
    w_y = (t - t0)/tE*np.sin(alpha) + u0*np.cos(alpha)
    w_points = w_x + 1j*w_y

    A = np.array([mag_vbb_binary(w0, rho, s, q) for w0 in w_points])

    M = np.stack([A - 1., np.ones_like(A)]).T
    beta = weighted_least_squares(fobs, ferr, M)
    fpred = (M @ beta).reshape(-1)

    if pointwise:
        return -0.5 * (fobs - fpred)**2 / ferr**2
    else:
        return -0.5 * np.sum((fobs - fpred)**2 / ferr**2)


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
t, fobs, ferr = np.array(t), np.array(fobs), np.array(ferr)


params_init = {
    'ln_s':np.log(2.218), 
    'ln_q':np.log(0.3392), 
    'ln_t0':np.log(7847.5), 
    'ln_u0':np.log(0.375), 
    'ln_tE':np.log(241.), 
    'alpha': 4.56 - np.pi ,
    'ln_rho':np.log(0.00097),
}
t0_est = np.exp(params_init['ln_t0'])



def compute_loglike_slice(
    npts,
    ln_s=None,
    ln_q=None,
    ln_t0=None,
    ln_u0=None,
    ln_tE=None,
    alpha=None,
    ln_rho=None,
):
    params = {
        'ln_s':ln_s if ln_s is not None else params_init['ln_s']*np.ones(npts),
        'ln_q':ln_q if ln_q is not None else params_init['ln_q']*np.ones(npts),
        'ln_t0':ln_t0 if ln_t0 is not None else params_init['ln_t0']*np.ones(npts),
        'ln_u0': ln_u0 if ln_u0 is not None else params_init['ln_u0']*np.ones(npts),
        'ln_tE': ln_tE if ln_tE is not None else params_init['ln_tE']*np.ones(npts),
        'alpha': alpha if alpha is not None else params_init['alpha']*np.ones(npts),
        'ln_rho': ln_rho if ln_rho is not None else params_init['ln_rho']*np.ones(npts),
    }
    # convert dictionary of arrays to list of dictionaries
    params_list = [dict(zip(params, t)) for t in zip(*params.values())]
    return np.array([ll_fn_vbb(p) for p in params_list])


def plot_loglike_slice(ax, xgrid, ygrid, vals, **kwargs):
    im = ax.pcolormesh(xgrid, ygrid, vals.reshape(xgrid.shape), cmap='turbo', zorder=-1, **kwargs)
    ix, iy = np.unravel_index(np.argmin(vals.reshape(-1)), xgrid.shape)
    return im, xgrid[ix, iy], ygrid[ix, iy]

# Load likelihood slices computed previously
dat = np.load(paths.data/"output/binary_lens/loglike_slices.npz")
q_grid, tE_grid, ll_q_tE, q_grid_zoom, tE_grid_zoom, ll_q_tE_zoom =\
     dat['q_grid'], dat['tE_grid'], dat['ll_grid_q_tE'], dat['q_grid_zoom'], dat['tE_grid_zoom'], dat['ll_grid_q_tE_zoom']
s_grid, u0_grid, ll_s_u0, s_grid_zoom, u0_grid_zoom, ll_s_u0_zoom =\
     dat['s_grid'], dat['u0_grid'], dat['ll_grid_s_u0'], dat['s_grid_zoom'], dat['u0_grid_zoom'], dat['ll_grid_s_u0_zoom']
alpha_grid, t0_grid, ll_alpha_t0, alpha_grid_zoom, t0_grid_zoom, ll_alpha_t0_zoom =\
     dat['alpha_grid'], dat['t0_grid'], dat['ll_grid_alpha_t0'], dat['alpha_grid_zoom'], dat['t0_grid_zoom'], dat['ll_grid_alpha_t0_zoom']

fig, ax = plt.subplots(
    8, 5, figsize=(12, 14),
    gridspec_kw={
    'height_ratios': [1, 3, 0.8, 1, 3,0.8, 1, 3], 
    'width_ratios': [3, 1, 0.8, 3, 1],
    'hspace': 0.05,
    'wspace':0.05,
    }
)

ax_main = [ax[1, 0], ax[4, 0], ax[7, 0]]
ax_zoom = [ax[1, 3], ax[4, 3], ax[7, 3]]
ax_slice_top = [ax[0, 0], ax[3, 0], ax[6, 0]]
ax_slice_side = [ax[1,1], ax[4, 1], ax[7, 1]]
ax_slice_top_zoom = [ax[0,3], ax[3, 3], ax[6, 3]]
ax_slice_side_zoom = [ax[1,4], ax[4, 4], ax[7, 4]]

for _a in ax_slice_top + ax_slice_top_zoom:
    _a.set(xticklabels=[], yticklabels=[])
for _a in ax_slice_side + ax_slice_side_zoom:
    _a.set(xticklabels=[], yticklabels=[])
for _a in ax[2, :]:
    _a.axis('off')
for _a in ax[5, :]:
    _a.axis('off')
for _a in ax[:, 2]:
    _a.axis('off')

for _a in [ax[0, 1], ax[3, 1], ax[6, 1]]:
    _a.axis('off')
for _a in [ax[0, 4], ax[3, 4], ax[6, 4]]:
    _a.axis('off')

for _a in [ax_main[0], ax_zoom[0]]:
    _a.set(xlabel='$q$')
for _a in [ax_main[1], ax_zoom[1]]:
    _a.set(xlabel='$s$')
for _a in [ax_main[2], ax_zoom[2]]:
    _a.set(xlabel='$\\alpha$')

ax_main[0].set_ylabel('$t_E$ [das]')
ax_main[1].set_ylabel('$u_0$')
ax_main[2].set_ylabel('$t_0$ [HJD - 2450000]')

vmin = 2_000
vmax = 120_000

# q vs tE
_, _, _ = plot_loglike_slice(
    ax_main[0],
    q_grid, 
    tE_grid,
    -ll_q_tE,
    vmin=vmin,
    vmax=vmax,
)
im, q_star, tE_star = plot_loglike_slice(
    ax_zoom[0],
    q_grid_zoom, 
    tE_grid_zoom,
    -ll_q_tE_zoom,
    vmin=vmin,
    vmax=vmax,
)
ax_main[0].axvline(q_star, color='white', lw=0.8)
ax_main[0].axhline(tE_star, color='white', lw=0.8)
ax_zoom[0].axvline(q_star, color='white', lw=0.8)
ax_zoom[0].axhline(tE_star, color='white', lw=0.8)


# s vs u0
_, _, _ = plot_loglike_slice(
    ax_main[1],
    s_grid,
    u0_grid,
    -ll_s_u0,
    vmin=vmin,
    vmax=vmax,
)
_, s_star, u0_star =plot_loglike_slice(
    ax_zoom[1],
    s_grid_zoom,
    u0_grid_zoom,
    -ll_s_u0_zoom,
    vmin=vmin,
    vmax=vmax,
)
ax_main[1].axvline(s_star, color='white', lw=0.8)
ax_main[1].axhline(u0_star, color='white', lw=0.8)
ax_zoom[1].axvline(s_star, color='white', lw=0.8)
ax_zoom[1].axhline(u0_star, color='white', lw=0.8)

# alpha vs t0
_, _, _ = plot_loglike_slice(
    ax_main[2],
    alpha_grid,
    t0_grid,
    -ll_alpha_t0,
    vmin=vmin,
    vmax=vmax,
)
_, alpha_star, t0_star = plot_loglike_slice(
    ax_zoom[2],
    alpha_grid_zoom,
    t0_grid_zoom,
    -ll_alpha_t0_zoom,
    vmin=vmin,
    vmax=vmax,
)
ax_main[2].axvline(alpha_star, color='white', lw=0.8)
ax_main[2].axhline(t0_star, color='white', lw=0.8)
ax_zoom[2].axvline(alpha_star, color='white', lw=0.8)
ax_zoom[2].axhline(t0_star, color='white', lw=0.8)


# 1D slices
npts_1D = 200

# q vs tE
q_slice = np.linspace(q_grid[0, 0], q_grid[0, -1], npts_1D)
ll_q = compute_loglike_slice(
    len(q_slice),
    ln_q = np.log(q_slice),
    ln_tE=np.log(tE_star)*np.ones_like(q_slice),
)

ax_slice_top[0].plot(q_slice, -ll_q, color='k', lw=1.5)
ax_slice_top[0].set(xlim=(q_slice[0], q_slice[-1]))

tE_slice = np.linspace(tE_grid[0, 0], tE_grid[-1, 0], npts_1D)
ll_tE = compute_loglike_slice(
    len(tE_slice),
    ln_q = np.log(q_star)*np.ones_like(tE_slice),
    ln_tE=np.log(tE_slice),
)

ax_slice_side[0].plot(-ll_tE, tE_slice, color='k', lw=1.5) 
ax_slice_side[0].set(ylim=(tE_slice[0], tE_slice[-1]))

# q vs tE zoom
q_slice_zoom = np.linspace(q_grid_zoom[0, 0], q_grid_zoom[0, -1], npts_1D)
ll_q_zoom = compute_loglike_slice(
    len(q_slice_zoom),
    ln_q = np.log(q_slice_zoom),
    ln_tE=np.log(tE_star)*np.ones_like(q_slice_zoom),
)

ax_slice_top_zoom[0].plot(q_slice_zoom, -ll_q_zoom, color='k', lw=1.5)
ax_slice_top_zoom[0].set(xlim=(q_slice_zoom[0], q_slice_zoom[-1]))

tE_slice_zoom = np.linspace(tE_grid_zoom[0, 0], tE_grid_zoom[-1, 0], npts_1D)
ll_tE_zoom = compute_loglike_slice(
    len(tE_slice_zoom),
    ln_q = np.log(q_star)*np.ones_like(tE_slice_zoom),
    ln_tE=np.log(tE_slice_zoom),
)

ax_slice_side_zoom[0].plot(-ll_tE_zoom, tE_slice_zoom, color='k', lw=1.5)
ax_slice_side_zoom[0].set(ylim=(tE_slice_zoom[0], tE_slice_zoom[-1]))

# s vs u0
s_slice = np.linspace(s_grid[0, 0], s_grid[0, -1], npts_1D)
ll_s = compute_loglike_slice(
    len(s_slice),
    ln_s = np.log(s_slice),
    ln_u0=np.log(u0_star)*np.ones_like(s_slice),
)

ax_slice_top[1].plot(s_slice, -ll_s, color='k', lw=1.5)
ax_slice_top[1].set(xlim=(s_slice[0], s_slice[-1]))

u0_slice = np.linspace(u0_grid[0, 0], u0_grid[-1, 0], npts_1D)
ll_u0 = compute_loglike_slice(
    len(u0_slice),
    ln_s = np.log(s_star)*np.ones_like(u0_slice),
    ln_u0=np.log(u0_slice),
)

ax_slice_side[1].plot(-ll_u0, u0_slice, color='k', lw=1.5)
ax_slice_side[1].set(ylim=(u0_slice[0], u0_slice[-1]))

# s vs u0 zoom
s_slice_zoom = np.linspace(s_grid_zoom[0, 0], s_grid_zoom[0, -1], npts_1D)
ll_s_zoom = compute_loglike_slice(
    len(s_slice_zoom),
    ln_s = np.log(s_slice_zoom),
    ln_u0=np.log(u0_star)*np.ones_like(s_slice_zoom),
)

ax_slice_top_zoom[1].plot(s_slice_zoom, -ll_s_zoom, color='k', lw=1.5)
ax_slice_top_zoom[1].set(xlim=(s_slice_zoom[0], s_slice_zoom[-1]))

u0_slice_zoom = np.linspace(u0_grid_zoom[0, 0], u0_grid_zoom[-1, 0], npts_1D)
ll_u0_zoom = compute_loglike_slice(
    len(u0_slice_zoom),
    ln_s = np.log(s_star)*np.ones_like(u0_slice_zoom),
    ln_u0=np.log(u0_slice_zoom),
)

ax_slice_side_zoom[1].plot(-ll_u0_zoom, u0_slice_zoom, color='k', lw=1.5)
ax_slice_side_zoom[1].set(ylim=(u0_slice_zoom[0], u0_slice_zoom[-1]))

# alpha vs t0
alpha_slice = np.linspace(alpha_grid[0, 0], alpha_grid[0, -1], npts_1D)
ll_alpha = compute_loglike_slice(
    len(alpha_slice),
    alpha= alpha_slice,
    ln_t0=np.log(t0_star)*np.ones_like(alpha_slice),
)

ax_slice_top[2].plot(alpha_slice, -ll_alpha, color='k', lw=1.5)
ax_slice_top[2].set(xlim=(alpha_slice[0], alpha_slice[-1]))

t0_slice = np.linspace(t0_grid[0, 0], t0_grid[-1, 0], npts_1D)
ll_t0 = compute_loglike_slice(
    len(t0_slice),
    alpha= alpha_star*np.ones_like(t0_slice),
    ln_t0=np.log(t0_slice),
)

ax_slice_side[2].plot(-ll_t0, t0_slice, color='k', lw=1.5)
ax_slice_side[2].set(ylim=(t0_slice[0], t0_slice[-1]))

# alpha vs t0 zoom
alpha_slice_zoom = np.linspace(alpha_grid_zoom[0, 0], alpha_grid_zoom[0, -1], npts_1D)
ll_alpha_zoom = compute_loglike_slice(
    len(alpha_slice_zoom),
    alpha = alpha_slice_zoom,
    ln_t0=np.log(t0_star)*np.ones_like(alpha_slice_zoom),
)

ax_slice_top_zoom[2].plot(alpha_slice_zoom, -ll_alpha_zoom, color='k', lw=1.5)
ax_slice_top_zoom[2].set(xlim=(alpha_slice_zoom[0], alpha_slice_zoom[-1]))

t0_slice_zoom = np.linspace(t0_grid_zoom[0, 0], t0_grid_zoom[-1, 0], npts_1D)
ll_t0_zoom = compute_loglike_slice(
    len(t0_slice_zoom),
    alpha= alpha_star*np.ones_like(t0_slice_zoom),
    ln_t0=np.log(t0_slice_zoom),
)

ax_slice_side_zoom[2].plot(-ll_t0_zoom, t0_slice_zoom, color='k', lw=1.5)
ax_slice_side_zoom[2].set(ylim=(t0_slice_zoom[0], t0_slice_zoom[-1]))

# custom axis for colorbar 
cax = fig.add_axes([0.7, 0.05, 0.2, 0.01])
cbar = fig.colorbar(im, cax=cax, orientation='horizontal', label='negative log likelihood')

for a in ax_main + ax_zoom:
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())

for _a in ax.reshape(-1):
    _a.set_rasterization_zorder(0)

fig.savefig(paths.figures/"binary_lens_likelihood_slices.pdf", bbox_inches="tight", dpi=200)