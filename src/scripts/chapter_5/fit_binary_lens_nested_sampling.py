import paths
import numpy as np
import os
from astropy.table import Table

from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import numpy as np

from caustics.linalg import weighted_least_squares

import ultranest

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

def prior_transform(u):
    """
    Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest.
    """
    x = np.array(u)  # copy u

    x[0] = np.log(7820.) + u[0] * (np.log(7860.) - np.log(7820.)) # ln_t0
    x[1] = np.log(10) + u[1] * (np.log(365.) - np.log(10.)) # ln_tE
    x[2] = np.log(0.001) + u[2] * (np.log(2.) - np.log(0.001)) # ln_u0
    x[3] = np.log(1e-02) + u[3] * (np.log(3.) - np.log(1e-02)) # ln_s
    x[4] = np.log(1e-05) + u[4] * (np.log(1.) - np.log(1e-05)) # ln_q
    x[5] = -np.pi + u[5] * (np.pi - (-np.pi)) # alpha
    x[6] = np.log(5e-04) + u[6] * (np.log(0.5) - np.log(5e-04)) # ln_rho

    return x
    
import VBBinaryLensing
VBBL = VBBinaryLensing.VBBinaryLensing()

def mag_vbb_binary(w0, rho, s, q, u1=0.0, accuracy=1e-05):
    return VBBL.BinaryMag2(s, q, w0.real, w0.imag, rho)

def log_likelihood(x):
    ln_t0, ln_tE, ln_u0, ln_s, ln_q, alpha, ln_rho = x
    s, q, t0, u0, tE, rho = np.exp(ln_s), np.exp(ln_q), np.exp(ln_t0),np.exp(ln_u0), np.exp(ln_tE), np.exp(ln_rho)

    w_x = (t - t0)/tE*np.cos(alpha) - u0*np.sin(alpha)
    w_y = (t - t0)/tE*np.sin(alpha) + u0*np.cos(alpha)
    w_points = w_x + 1j*w_y

    A = np.array([mag_vbb_binary(w0, rho, s, q) for w0 in w_points])

    M = np.stack([A - 1., np.ones_like(A)]).T
    beta = np.array(weighted_least_squares(fobs, ferr, M))
    fpred = (M @ beta).reshape(-1)

    return -0.5 * np.sum((fobs - fpred)**2 / ferr**2)

# STEPSAMPLER
import ultranest.stepsampler
param_names = ['ln_t0', 'ln_tE', 'ln_u0', 'ln_s', 'ln_q', 'alpha', 'ln_rho']
sampler = ultranest.ReactiveNestedSampler(
    param_names,
    log_likelihood,
    prior_transform,
    resume='resume-similar',
    log_dir='../data/output/binary_lens/ultranest',

)

nsteps = 100
# create step sampler:
sampler.stepsampler = ultranest.stepsampler.SliceSampler(
    nsteps=nsteps,
    generate_direction=ultranest.stepsampler.generate_mixture_random_direction,
    # max_nsteps=400
)

result = sampler.run(min_num_live_points=1000)
