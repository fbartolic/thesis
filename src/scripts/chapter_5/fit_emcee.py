import numpy as np
import os
from astropy.table import Table

import emcee
from caustics.linalg import weighted_least_squares

import arviz as az 
os.environ["OMP_NUM_THREADS"] = "1"

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


lc = get_data("../../../../../data.nosync/OGLE_ews/2017/blg-0039/")

# Trim
lc['HJD'] = lc['HJD'] - 2450000
lc = lc[lc['HJD'] > 7400]
fobs, ferr = magnitudes_to_fluxes(lc["mag"], lc["mag_err"])
t = lc['HJD']
t, fobs, ferr = np.array(t), np.array(fobs), np.array(ferr)

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
    beta = weighted_least_squares(fobs, ferr, M)
    fpred = (M @ beta).reshape(-1)

    return -0.5 * np.sum((fobs - fpred)**2 / ferr**2)

def log_prior(x):
    ln_t0, ln_tE, ln_u0, ln_s, ln_q, alpha, ln_rho = x
    if not (np.log(7830.) < ln_t0 < np.log(7835.)):
        return -np.inf
    if not (np.log(140.) < ln_tE < np.log(170.)):
        return -np.inf
    if not (np.log(0.0005) < ln_u0 < np.log(0.05)):
        return -np.inf
    if not (np.log(1.5) < ln_s < np.log(1.7)):
        return -np.inf
    if not (np.log(0.15) < ln_q < np.log(0.25)):
        return -np.inf
    if not (-1.4 < alpha < -1.1):
        return -np.inf
    if not (np.log(5e-04) < ln_rho < np.log(5e-03)):
        return -np.inf
    return 0.0

def log_prob(x):
    return  log_prior(x) + log_likelihood(x)

param_names = ['ln_t0', 'ln_tE', 'ln_u0', 'ln_s', 'ln_q', 'alpha', 'ln_rho']
def get_samples_from_ns(path):
    chains_equal_weighted = np.loadtxt(path, skiprows=1)[-10000:, :]
    samples_ns = {name: chains_equal_weighted[-5000:, i] for i, name in enumerate(param_names)}
    samples_ns_az = az.from_dict(samples_ns)
    return samples_ns_az

samples_ns_az_mode1 = get_samples_from_ns("../../data/output/binary_lens/nlive_2000_mode1/chains/equal_weighted_post.txt")


from multiprocessing import Pool

ndim, nwalkers = 7, 50
p0 = samples_ns_az_mode1.posterior.to_array().values[:, 0, :nwalkers].T

nwarmup, nsteps = 5000, 50000

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
    state = sampler.run_mcmc(p0, nwarmup, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=True)


samples_az = az.from_emcee(sampler, var_names=param_names)

az.to_netcdf(samples_az, "../../data/output/binary_lens/emcee_50000_5000_50_7.nc")