import paths
import numpy as np

import os
from astropy.table import Table

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

# Load light curve
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
    im = ax.pcolormesh(xgrid, ygrid, vals, cmap='turbo', **kwargs)
    ix, iy = np.unravel_index(np.argmin(vals.reshape(-1)), xgrid.shape)
    ax.scatter(xgrid[ix, iy], ygrid[ix, iy], marker='x', color='white', s=100)
    return im

# PLOT 1: s vs u0
ngrid = 250
_s_linspace = np.linspace(0.1, 3., ngrid)
_u0_linspace = np.linspace(0.01, 0.7, ngrid)
s_grid, u0_grid = np.meshgrid(_s_linspace, _u0_linspace)

ll_grid_s_u0 = compute_loglike_slice(
    len(s_grid.reshape(-1)), 
    ln_s=np.log(s_grid.reshape(-1)), 
    ln_u0=np.log(u0_grid.reshape(-1)),
)

# zoom in
_s_linspace_zoom = np.linspace(1.9, 2.4, ngrid)
_u0_linspace_zoom = np.linspace(0.28, 0.45, ngrid)
s_grid_zoom, u0_grid_zoom = np.meshgrid(_s_linspace_zoom, _u0_linspace_zoom)

ll_grid_s_u0_zoom = compute_loglike_slice(
    len(s_grid_zoom.reshape(-1)), 
    ln_s=np.log(s_grid_zoom.reshape(-1)), 
    ln_u0=np.log(u0_grid_zoom.reshape(-1)),
)

# Plot 2: alpha vs t0 
_alpha_linspace = np.linspace(-np.pi, np.pi, ngrid)
_t0_linspace = np.linspace(7820, 7860, ngrid)
alpha_grid, t0_grid = np.meshgrid(_alpha_linspace, _t0_linspace)

ll_grid_alpha_t0 = compute_loglike_slice(
    len(alpha_grid.reshape(-1)), 
    ln_t0=np.log(t0_grid.reshape(-1)), 
    alpha=alpha_grid.reshape(-1),
)

# zoom in
_alpha_linspace_zoom = np.linspace(0.9, 1.8, ngrid)
_t0_linspace_zoom = np.linspace(7845, 7855, ngrid)
alpha_grid_zoom, t0_grid_zoom = np.meshgrid(_alpha_linspace_zoom, _t0_linspace_zoom)

ll_grid_alpha_t0_zoom = compute_loglike_slice(
    len(alpha_grid_zoom.reshape(-1)),
    ln_t0=np.log(t0_grid_zoom.reshape(-1)),
    alpha=alpha_grid_zoom.reshape(-1),
)

# Plot 3: q vs tE
_q_linspace = np.linspace(0.2, 0.5, ngrid)
_tE_linspace = np.linspace(200, 300, ngrid)
q_grid, tE_grid = np.meshgrid(_q_linspace, _tE_linspace)

ll_grid_q_tE = compute_loglike_slice(
    len(q_grid.reshape(-1)),
    ln_tE=np.log(tE_grid.reshape(-1)),
    ln_q=np.log(q_grid.reshape(-1)),
)

# zoom in
_q_linspace_zoom = np.linspace(0.3, 0.366, ngrid)
_tE_linspace_zoom = np.linspace(225, 250, ngrid)
q_grid_zoom, tE_grid_zoom = np.meshgrid(_q_linspace_zoom, _tE_linspace_zoom)

ll_grid_q_tE_zoom = compute_loglike_slice(
    len(q_grid_zoom.reshape(-1)),
    ln_tE=np.log(tE_grid_zoom.reshape(-1)),
    ln_q=np.log(q_grid_zoom.reshape(-1)),
)

# Save to file 
np.savez(
    paths.data/'output/binary_lens/loglike_slices.npz',
    s_grid=s_grid, u0_grid=u0_grid, ll_grid_s_u0=ll_grid_s_u0,
    s_grid_zoom=s_grid_zoom, u0_grid_zoom=u0_grid_zoom, ll_grid_s_u0_zoom=ll_grid_s_u0_zoom,
    alpha_grid=alpha_grid, t0_grid=t0_grid, ll_grid_alpha_t0=ll_grid_alpha_t0,
    alpha_grid_zoom=alpha_grid_zoom, t0_grid_zoom=t0_grid_zoom, ll_grid_alpha_t0_zoom=ll_grid_alpha_t0_zoom,
    q_grid=q_grid, tE_grid=tE_grid, ll_grid_q_tE=ll_grid_q_tE,
    q_grid_zoom=q_grid_zoom, tE_grid_zoom=tE_grid_zoom, ll_grid_q_tE_zoom=ll_grid_q_tE_zoom,
)
