import paths
from functools import partial
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import random, vmap, jit
import jax.scipy as jsp

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

from caustics.linalg import weighted_least_squares 
from caustics.trajectory import AnnualParallaxTrajectory

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from scipy.signal import medfilt

import numpyro
import numpyro.distributions as dist
from numpyro.infer.util import  log_likelihood, init_to_value
from numpyro.infer import *
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.infer.elbo import Trace_ELBO
numpyro.set_host_device_count(4)

from numpyro_ext.optim import optimize as optimize_numpyro_ext

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
    A_u0 = (u0 ** 2 + 2) / (jnp.abs(u0) * jnp.sqrt(u0 ** 2 + 4))
    A = (A - 1) / (A_u0 - 1)

    M = jnp.stack([A, jnp.ones_like(A)]).T
    beta = weighted_least_squares(fobs, ferr, M)
    fpred = (M @ beta).reshape(-1)

    numpyro.sample("obs", dist.Normal(fpred, ferr), obs=fobs)

def initialization_model(t0_est):
    t0 = numpyro.sample('t0', dist.Normal(t0_est, 10.))
    numpyro.deterministic('ln_t0', jnp.log(t0))
    tE = numpyro.sample('tE', dist.Uniform(10., 400.))
    numpyro.deterministic('ln_tE', jnp.log(tE))
    numpyro.sample('u0', dist.Uniform(-1., 1.))
    numpyro.sample('piEE', dist.Normal(0., 0.1))
    numpyro.sample('piEN', dist.Normal(0., 0.1))


def fit_map(key, t0_est, params_init):
    opt = optimize_numpyro_ext(model, start=params_init, num_steps=3, include_deterministics=False)
    return opt(key, t=t, t0_est=t0_est, fobs=fobs, ferr=ferr, trajectory=trajectory)

def fit_map_batched(rng_key, params_init_list):
    rng_key, subkey = random.split(rng_key)
    rng_keys = random.split(subkey, len(params_init_list['u0']))

    map_params_list = []
    for i, key in enumerate(rng_keys):
        params_init = {k: v[i] for k, v in params_init_list.items()}
        map_params = fit_map(key, t0_est, params_init)
        map_params_list.append(map_params)
    map_params_list = {k: jnp.stack([p[k] for p in map_params_list]) for k in map_params_list[0].keys()}

    f = lambda params: log_likelihood(
        model,
        params, 
        t=t, t0_est=t0_est, fobs=fobs,ferr=ferr, trajectory=trajectory
    )['obs'].sum()

    ll_list = jnp.round(vmap(f)(map_params_list),1)
    _, idcs_unique = jnp.unique(ll_list, return_index=True)

    map_params_list = {k: v[idcs_unique] for k, v in map_params_list.items()}

    return map_params_list 

@jit
def get_laplace_posterior(map_params_constrained):
    # Evaluate logp for the Laplace posterior approximation (MVN distribution)
    # for each sample
    guide = AutoLaplaceApproximation(model, init_loc_fn=init_to_value(values=map_params_constrained))
    svi = SVI(model, guide, numpyro.optim.Minimize(method='BFGS'), Trace_ELBO())
    state = svi.init(
        random.PRNGKey(0), t=t, t0_est=t0_est, fobs=fobs, ferr=ferr, trajectory=trajectory, 
    )
    _params_guide_unconstrained = svi.get_params(state)
    post = guide.get_posterior(_params_guide_unconstrained)
    return post 

 
@partial(jit, static_argnames=("nsamples",))
def get_laplace_posterior_samples(map_params_constrained, nsamples=5000):
    guide = AutoLaplaceApproximation(model, init_loc_fn=init_to_value(values=map_params_constrained))
    svi = SVI(model, guide, numpyro.optim.Minimize(method='BFGS'), Trace_ELBO())
    state = svi.init(
        random.PRNGKey(0), t=t, t0_est=t0_est, fobs=fobs, ferr=ferr, trajectory=trajectory
    )
    params_guide_unconstrained = svi.get_params(state)
    post = guide.get_posterior(params_guide_unconstrained)
    unconstrained_samples = post.sample(random.PRNGKey(0), sample_shape=(nsamples,))
    constrained_samples = guide._unpack_and_constrain(unconstrained_samples, params_guide_unconstrained)
    unconstrained_samples = {key:unconstrained_samples[:, i] for i, key in enumerate(map_params_constrained.keys())}
    return constrained_samples, unconstrained_samples

   
@jit
def compute_loo_log_weights(samples_constrained, samples_unconstrained, map_params_constrained):
    # Evaluate the pointwise log-likelihood for each sample from Laplace guide 
    ll_pointwise = log_likelihood(
        model,
        samples_constrained, 
        t=t, t0_est=t0_est, fobs=fobs,ferr=ferr, trajectory=trajectory
    )['obs']

    # Evaluate model logp in the *unconstrained parameters* for each
    # sample from Laplace guide
    log_prob = lambda params_unconstrained: -numpyro.infer.util.potential_energy(
        model,
        model_args=(t, t0_est, fobs, ferr, trajectory),
        model_kwargs={},
        params=params_unconstrained,
    )
    logps = vmap(log_prob)(samples_unconstrained)

    # Evaluate logp for the Laplace posterior approximation (MVN distribution)
    post = get_laplace_posterior(map_params_constrained)
    logps_guide = vmap(post.log_prob)(jnp.stack(list(samples_unconstrained.values())).T)

    # Compute the (log) importance sampling weights from https://proceedings.mlr.press/v97/magnusson19a.html
    log_weights = -ll_pointwise + logps[:, None] - logps_guide[:, None]

    return log_weights, ll_pointwise


def compute_loo(log_weights, ll_pointwise):
    from arviz.stats.stats_utils import ELPDData
    import xarray as xr
    n_samples, n_data_points = log_weights.shape

    # Pareto smoothing
    log_weights, pareto_shape = az.psislw(np.array(log_weights.T))
    log_weights = log_weights.T

    # Pointwise loo_lppd
    loo_lppd_i = jsp.special.logsumexp(ll_pointwise + log_weights, axis=0) - jsp.special.logsumexp(log_weights, axis=0)

    # loo_lppd estimate and standard error
    loo_lppd = jnp.sum(loo_lppd_i)
    loo_lppd_se = jnp.sqrt(loo_lppd_i.shape[0])*jnp.std(loo_lppd_i)

    # Effective number of parameters
    lppd = jnp.sum(jsp.special.logsumexp(ll_pointwise, axis=0) - jnp.log(ll_pointwise.shape[0]))
    p_loo = lppd - loo_lppd 

    return ELPDData(
        data=[
            float(loo_lppd),
            float(loo_lppd_se),
            float(p_loo),
            n_samples,
            n_data_points,
            False,
            xr.DataArray(data=np.array(loo_lppd_i), name='loo_i'),
            xr.DataArray(data=np.array(pareto_shape), name='pareto_shape'),
            "log",
        ],
        index=[
            "loo",
            "loo_se",
            "p_loo",
            "n_samples",
            "n_data_points",
            "warning",
            "loo_i",
            "pareto_k",
            "loo_scale",
        ],
    )


lc = get_data(paths.data/"microlensing/starBLG234.6.I.218982.dat")

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


# Fit Laplace approximation at multiple starting points
key = random.PRNGKey(0)
key, subkey = random.split(key)

prior_predictive = Predictive(initialization_model, num_samples=10)
samples_init = prior_predictive(subkey, t0_est=t0_est)

key, subkey = random.split(key)
map_params_dict = fit_map_batched(subkey, samples_init)
map_params_dict = {key: map_params_dict[key][::-1] for key in map_params_dict.keys()}
params_names = list(map_params_dict.keys())

# Get samples from Laplace approximation  
samples_laplace_dict, samples_laplace_unconstrained_dict = vmap(get_laplace_posterior_samples)(map_params_dict)
samples_laplace_az = az.from_dict(samples_laplace_dict)


# Compute loo weights
log_weights_list, ll_pointwise_list =\
    vmap(compute_loo_log_weights)(samples_laplace_dict, samples_laplace_unconstrained_dict, map_params_dict)


# Compute loo for each mode
compare_dict_laplace = {
    f'mode_{i+1}': compute_loo(log_weights_list[i], ll_pointwise_list[i]) for i in range(len(log_weights_list))
}
elpd_data_list = [compute_loo(log_weights_list[i], ll_pointwise_list[i]) for i in range(len(log_weights_list))]

# Sort according to loo and keep the significant modes
loo_vals = np.array([e['loo'] for e in elpd_data_list])
mask = np.nanmax(loo_vals) - loo_vals < 50
idcs = np.where(mask)[0]
samples_laplace_az = samples_laplace_az.isel(chain=idcs)
compare_dict_laplace = {f'mode_{i+1}': elpd_data_list[j] for i, j in enumerate(idcs)}

print(az.compare(compare_dict_laplace))

# Load samples from disk with arviz
samples_mcmc_az = az.from_netcdf(paths.scripts/"output/single_lens/samples_mcmc_az.nc")
samples_mcmc_az.posterior['piE'] = np.sqrt(samples_mcmc_az.posterior.piEE**2 + samples_mcmc_az.posterior.piEN**2)
compare_dict_mcmc = {
    f'mode_{i}': samples_mcmc_az.isel(chain=[i]) for i in range(4)
}

az.compare(compare_dict_mcmc, method='BB-pseudo-BMA')


# Plot NUTS and Laplace samples
fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'wspace': 0.2})

for i in range(4):
    _samples = samples_mcmc_az.isel(chain=[i]).stack(sample=('chain', 'draw')).posterior
    _samples_laplace = samples_laplace_az.isel(chain=[i]).stack(sample=('chain', 'draw')).posterior

    ax[0].scatter(
        jnp.exp(_samples['ln_tE'].values),
        _samples['piE'].values,
        color=f'C{i}', alpha=0.1, label=f'Mode {i + 1}',
        zorder=-1,
    )
    ax[1].scatter(
        jnp.exp(_samples_laplace['ln_tE'].values),
        _samples_laplace['piE'].values,
        color=f'C{i}', alpha=0.1, label=f'Mode {i + 1}', marker='o',
        zorder=-1,
    )

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='C0', label='Mode 1',ls=''),
    Line2D([0], [0], marker='o', color='C1', label='Mode 2',ls=''),
    Line2D([0], [0], marker='o', color='C2', label='Mode 3',ls=''),
    Line2D([0], [0], marker='o', color='C3', label='Mode 4',ls=''),
]

ax[1].legend(handles=legend_elements, fontsize=14)

for _a in ax:
    _a.set(xlabel=r'$t_E$ [days]', ylabel=r'$\pi_E$ ');
    _a.set(xlim=(80, 190), ylim=(0.15, 0.45))
    _a.set_rasterization_zorder(0)

ax[0].set_title("NUTS HMC samples")
ax[1].set_title("Laplace approximation samples")

for _a in ax:
    _a.xaxis.set_minor_locator(AutoMinorLocator())
    _a.yaxis.set_minor_locator(AutoMinorLocator())

fig.savefig(paths.figures/"single_lens_samples_nuts_vs_laplace.pdf", bbox_inches="tight")


# Compare ECDFs
fig, ax = plt.subplots(2, 4, figsize=(16, 8),
gridspec_kw={'wspace':0.3, 'hspace':0.3})

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y


for i in range(4):
    _samples = samples_mcmc_az.isel(chain=[i]).stack(sample=('chain', 'draw')).posterior
    _samples_laplace = samples_laplace_az.isel(chain=[i]).stack(sample=('chain', 'draw')).posterior

    nthin = 3
    x1, y1 = ecdf(np.exp(_samples['ln_tE'][::nthin].values))
    x2, y2 = ecdf(np.exp(_samples_laplace['ln_tE'].values[:len(x1)]))
    ax[0, i].plot(x1, y1, 'k.', alpha=0.5, label='NUTS', zorder=-1)
    ax[0, i].plot(x2, y2, 'C1.', alpha=0.5, label='Laplace', zorder=-1)


    x1, y1 = ecdf(_samples['piE'][::4].values)
    x2, y2 = ecdf(_samples_laplace['piE'].values[:len(x1)])
    ax[1, i].plot(x1, y1, 'k.', alpha=0.5, zorder=-1)
    ax[1, i].plot(x2, y2, 'C1.', alpha=0.5, zorder=-1)

    k_hat_mean = np.round(np.mean(compare_dict_laplace[f'mode_{i+1}'].pareto_k.values),2)
    ax[0, i].set_title(f"Mode {i+1}, $\mathrm{{mean}}(\hat k)={k_hat_mean}$" )
    ax[0, i].set_xlabel("$t_E$ [days]")
    ax[1, i].set_xlabel("$\pi_E$")
    ax[0, i].grid(alpha=0.5)
    ax[1, i].grid(alpha=0.5)

ax[0, 0].set_ylabel("ECDF")
ax[1, 0].set_ylabel("ECDF")
ax[0, -1].legend(fontsize=14)

for _a in ax.reshape(-1):
    _a.xaxis.set_minor_locator(AutoMinorLocator())
    _a.yaxis.set_minor_locator(AutoMinorLocator())
    _a.set_rasterization_zorder(0)

fig.savefig(paths.figures/"single_lens_samples_nuts_vs_laplace_ecdf.pdf", bbox_inches="tight")