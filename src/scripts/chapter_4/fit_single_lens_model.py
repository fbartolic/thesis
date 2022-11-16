import paths
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
import jax.scipy as jsp

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

from caustics.linalg import weighted_least_squares 
from caustics.trajectory import AnnualParallaxTrajectory

from scipy.signal import medfilt

import ultranest

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer import *
numpyro.set_host_device_count(4)

import arviz as az

def magnitudes_to_fluxes(mag, mag_err, zeropoint=22):
    flux = 10. ** (0.4 * (zeropoint - mag))
    err_flux = mag_err * flux * np.log(10.) * 0.4
    return flux, err_flux

def get_data(path_to_file):
    t = Table.read(path_to_file, format="ascii")

    # Remove additional columns
    t.columns[0].name = "HJD"
    t.columns[1].name = "mag"
    t.columns[2].name = "mag_err"
    t.keep_columns(("HJD", "mag", "mag_err"))
    return t

lc = get_data("../data/microlensing/starBLG234.6.I.218982.dat")

# Trim
t = lc['HJD']
fobs, ferr = magnitudes_to_fluxes(lc["mag"], lc["mag_err"])
t, fobs, ferr = jnp.array(t), jnp.array(fobs), jnp.array(ferr)

RA = "18:04:45.70"
Dec = "-26:59:15.5"
coords = SkyCoord(
    RA, Dec, unit=(u.hourangle, u.deg, u.arcminute)
)
trajectory = AnnualParallaxTrajectory(t, coords)
t0_est = t[np.argmax(medfilt(fobs, 9))]

def model(t, t0_est, fobs, ferr, trajectory):
    ln_t0 = numpyro.sample("ln_t0", dist.Normal(jnp.log(t0_est), jnp.log(10.)))
    ln_tE = numpyro.sample("ln_tE", dist.Normal(2.5, 1.5))
    u0 = numpyro.sample("u0", dist.Normal(0.0, 1.))
    piEE = numpyro.sample("piEE", dist.Normal(0.0, 1.))
    piEN = numpyro.sample("piEN", dist.Normal(0.0, 1.))
    numpyro.deterministic("piE", jnp.sqrt(piEE**2 + piEN**2))
    t0, tE = jnp.exp(ln_t0), jnp.exp(ln_tE)
    
    # Compute trajectory
    w_points = trajectory.compute(
        t, t0=t0, tE=tE, u0=u0, piEE=piEE, piEN=piEN
    )
    u = jnp.abs(w_points)
    A = (u ** 2 + 2) / (u * jnp.sqrt(u ** 2 + 4))

    M = jnp.stack([A - 1., jnp.ones_like(A)]).T
    beta = weighted_least_squares(fobs, ferr, M)
    fpred = (M @ beta).reshape(-1)

    numpyro.deterministic("F_s", beta.reshape(-1)[0])
    numpyro.deterministic("F_base", beta.reshape(-1)[1])

    numpyro.sample("obs", dist.Normal(fpred, ferr), obs=fobs)


def fit_mcmc_numpyro(key, model, start, n_warmup=1000, n_samples=2000, n_chains=1):
    mcmc = MCMC(
        NUTS(model, init_strategy=numpyro.infer.util.init_to_value(values=start), target_accept_prob=0.95),
        num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains,
        progress_bar=False
    )
    key, subkey = random.split(key)
    mcmc.run(key, t, t0_est, fobs, ferr, trajectory)

    samples = mcmc.get_samples(group_by_chain=False)
    ll_pointwise = numpyro.infer.log_likelihood(
        model, samples, t=t, t0_est=t0_est, fobs=fobs, ferr=ferr, trajectory=trajectory
    )['obs']

    return  samples, ll_pointwise

start1 = {'ln_t0': 8.196958674800106,
 'ln_tE': 4.679257347077028,
 'piEE': 0.12107727135943634,
 'piEN': -0.314272509516472,
 'u0': -0.4330639292916798}

start2 = {'ln_t0': 8.19693545922209,
 'ln_tE': 4.52023750854379,
 'piEE': 0.1120929972449304,
 'piEN': 0.24743514285471713,
 'u0': 0.4849923706653194}

start3 = {'ln_t0': 8.197058777217139,
 'ln_tE': 4.795905185268507,
 'piEE': 0.0773065780976567,
 'piEN': 0.3319138992567253,
 'u0': -0.23152238140113068}

start4 = {'ln_t0': 8.197108205402651,
 'ln_tE': 5.015234049929733,
 'piEE': 0.07897808634119093,
 'piEN': -0.2988402065893234,
 'u0': 0.17898376720488435}

params_names = list(start1.keys())

key = random.PRNGKey(0)
key, subkey = random.split(key)

# NUTS fit
initial_positions = {key:jnp.array([start1[key], start2[key], start3[key], start4[key]]) for key in start1.keys()}

keys = random.split(random.PRNGKey(0), 4)
samples_list, ll_pointwise_list = pmap(
    lambda key, start: fit_mcmc_numpyro(key, model, start, n_warmup=500, n_samples=3000, n_chains=1)
)(keys, initial_positions)
samples_list

def convert_to_az(samples, ll_pointwise):
    samples_az = az.convert_to_inference_data(samples)
    samples_az.add_groups(
        log_likelihood={"y":ll_pointwise[None, :, :]},
    )
    return samples_az

samples_mcmc_az_list = [
    convert_to_az({key:samples_list[key][i] for key in samples_list.keys()}, ll_pointwise_list[i]) for i in range(4)
]
samples_mcmc_az = az.concat(samples_mcmc_az_list, dim='chain')

# Save to disk 
az.to_netcdf(samples_mcmc_az, paths.data/'output/single_lens/samples_mcmc_az.nc')

# NESTED SAMPLING
@jit
def prior_transform(u):
    """
    Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest.
    """
    x = jnp.array(u)  # copy u

    # ln_t0
    x = x.at[0].set(jsp.stats.norm.ppf(u[0], jnp.log(t0_est), jnp.log(10)))

    # ln_tE
    x = x.at[1].set(jsp.stats.norm.ppf(u[1], 2.5, 1.5))
#    x = x.at[1].set(
#        jnp.log(10) + u[1] * (jnp.log(365.) - jnp.log(10.))
#    )
    # u0
    x = x.at[2].set(jsp.stats.norm.ppf(u[2], 0, 1.))

    # pi_EE
    x = x.at[3].set(jsp.stats.norm.ppf(u[3], 0., 1.))

    # pi_EN
    x = x.at[4].set(jsp.stats.norm.ppf(u[4], 0., 1.))

    return x

log_likelihood_fn = lambda x: numpyro.infer.log_likelihood(
    model,
    {'ln_t0': x[0], 'ln_tE': x[1], 'u0': x[2], 'piEE': x[3], 'piEN': x[4]},
    t=t, t0_est=t0_est, fobs=fobs,ferr=ferr, trajectory=trajectory
)['obs'].sum()
log_likelihood_fn = jit(log_likelihood_fn)

log_likelihood_vectorized = lambda x: np.array(vmap(log_likelihood_fn)(x))
prior_transform_vectorized = lambda x: np.array(vmap(prior_transform)(x))

param_names = ['ln_t0', 'ln_tE', 'u0', 'pi_EE', 'pi_EN']
sampler = ultranest.ReactiveNestedSampler(
    param_names,
    log_likelihood_vectorized,
    prior_transform_vectorized, 
    resume=True,
    vectorized=True,
    log_dir=paths.data/'output/single_lens/ultranest',
);
result = sampler.run()

