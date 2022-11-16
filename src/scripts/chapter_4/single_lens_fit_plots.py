import paths
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import random

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

from caustics.linalg import weighted_least_squares 
from caustics.trajectory import AnnualParallaxTrajectory

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import medfilt

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


def predict_flux(t, t_dense, samples, nsamples=20):
    fpred_list = []
    fpred_dense_list = []
    w_list = []

    for i in random.randint(random.PRNGKey(0), (nsamples,), 0, samples['u0'].values.shape[0]):
        # Compute trajectory
        w = trajectory.compute(
            t, t0=jnp.exp(samples['ln_t0'].values[i]), tE=jnp.exp(samples['ln_tE'].values[i]), 
            u0=samples['u0'].values[i], piEE=samples['piEE'].values[i], piEN=samples['piEN'].values[i]
        )
        w_dense = trajectory.compute(
            t_dense, t0=jnp.exp(samples['ln_t0'].values[i]), tE=jnp.exp(samples['ln_tE'].values[i]), 
            u0=samples['u0'].values[i], piEE=samples['piEE'].values[i], piEN=samples['piEN'].values[i]
        )
 
        u = jnp.abs(w)
        A = (u ** 2 + 2) / (u * jnp.sqrt(u ** 2 + 4))

        u_dense = jnp.abs(w_dense)
        A_dense = (u_dense ** 2 + 2) / (u_dense * jnp.sqrt(u_dense ** 2 + 4))

        M = jnp.stack([A - 1., jnp.ones_like(A)]).T
        beta = weighted_least_squares(fobs, ferr, M)
        M_dense = jnp.stack([A_dense - 1., jnp.ones_like(A_dense)]).T

        fpred = (M @ beta).reshape(-1)
        fpred_dense = (M_dense @ beta).reshape(-1)
        fpred_list.append(fpred)
        fpred_dense_list.append(fpred_dense)
        w_list.append(w_dense)

    return jnp.stack(fpred_list), jnp.stack(fpred_dense_list), jnp.stack(w_list)

def mixture_draws(samples, weights, nthin=None, permutation=True):
    """
    Given samples from K MCMC chains and weights w_k, k=1..K, the function returns
    nthin simulation draws approximating the weighted mixture of K distributions.
    
    Adapted from https://github.com/yao-yl/Multimodal-stacking-code/blob/master/chain_stacking.R
    
    Args:
        samples (ndarray): Parameter samples from K chains, shape (nsamples, K).
        weights (ndarray): Size K array of weights.
        nthin (int): Number of samples after thining, if None samples won't be thinned. Defaults to None.
        permutation (bool): Randomly permute the samples. Defaults to True.
    Returns:
        ndarray: Samples from the mixture distribution, shape (nvar, nsamples).
    """
    dct = {}
    
    for varname in samples.keys():
        s = samples[varname].values.T
        nsamples = s.shape[0]
        K = s.shape[1]
        
        if nthin is None:
            nthin = nsamples

        if permutation is True:
            s = np.random.permutation(s)

        integer_part = np.floor(nthin*weights).astype(np.int32)
        existing_draws = int(sum(integer_part))

        if existing_draws < nthin:
            remaining_draws = nthin - existing_draws
            update_w = (weights - integer_part/nthin)*nthin/remaining_draws
            remaining_assignment = np.random.choice(K, remaining_draws, p=update_w, replace=False)
            integer_part[remaining_assignment] = integer_part[remaining_assignment] + 1
        

        integer_part_index = np.insert(np.cumsum(integer_part), 0, 0, axis=0).astype(np.int32)
        mixture_vector = np.zeros(nthin)

        for k in range(K):
            if(1 + integer_part_index[k]) <= integer_part_index[k + 1]:
                mixture_vector[integer_part_index[k]:integer_part_index[k + 1]] = s[:integer_part[k], k]
                
        dct[varname] = mixture_vector
        
    return dct



# LOAD DATA
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

# PLOT LIGHTCURVE
fig, ax = plt.subplots(figsize=(12, 5))
ax.errorbar(lc['HJD'], lc['mag'], lc['mag_err'], fmt="o", alpha=0.3, color='k')
# invert y axis
ax.invert_yaxis()
ax.set(xlabel='Time [HJD - 2450000 days]', ylabel='I magnitude')
ax.grid(alpha=0.5)
ax.set_title("OGLE-2005-BLG-086")
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
fig.savefig(paths.figures/"single_lens_lightcurve.pdf", bbox_inches="tight")


# Load samples from disk with arviz
samples_mcmc_az = az.from_netcdf(paths.data/"output/single_lens/samples_mcmc_az.nc")
samples_mcmc_az.posterior['piE'] = np.sqrt(samples_mcmc_az.posterior.piEE**2 + samples_mcmc_az.posterior.piEN**2)


# Print Quantiles
samples_mcmc_az.posterior['t0'] = np.exp(samples_mcmc_az.posterior.ln_t0)
samples_mcmc_az.posterior['tE'] = np.exp(samples_mcmc_az.posterior.ln_tE)

for i in range(4):
    print(f"Mode {i} Inferred parameters (0.15, 0.5, 0.85) quantiles:\n")
    quantiles = samples_mcmc_az.posterior[['t0', 'tE', 'u0', 'piEN', 'piEE']].isel(
        chain=[i]
    ).quantile((0.15, 0.5, 0.85), dim='draw').drop(['chain', 'quantile'])

    for var in quantiles.variables:
        vals = quantiles[var].values.reshape(-1)

        # same statement but using f-strings and :.2f for formatting
        print(f"{var} = {vals[1]:.3f}_{{-{vals[1] - vals[0]:.3f}}}^{{+{vals[2] - vals[1]:.3f}}}")
    print("\n\n")


chi_sq_vals = -2*(samples_mcmc_az.log_likelihood.mean(dim='draw').sum(dim='y_dim_0').y.values +\
     0.5*jnp.sum(jnp.log(2*jnp.pi*ferr**2)))
chi_sq_vals = {f"mode{i+1}": chi_sq_vals[i] for i in range(4)}
print("chi squared mean values: ", chi_sq_vals)

# Posterior predictions in data space
fpred_dict = {}
fpred_dense_dict = {}
w_dense_dict = {}
t_dense = jnp.linspace(t[0], t[-1], 2000)

for i, key in enumerate(['mode1', 'mode2', 'mode3', 'mode4']):
    samples = samples_mcmc_az.isel(chain=[i]).stack(sample=('chain', 'draw')).posterior
    fpred_dict[key], fpred_dense_dict[key], w_dense_dict[key] = predict_flux(
        t, t_dense, samples, nsamples=20
    )

# make plot
fig, ax = plt.subplots(
    3, 2, figsize=(16, 10), sharex=True, 
    gridspec_kw={'wspace':0.05}
)
for i, key in enumerate(['mode1', 'mode4']):
    j = int(key[-1])
    ax[0, i].errorbar(t - 3620., fobs, ferr,  marker='o', color='k', linestyle='None', alpha=0.2, zorder=-1)

    for s in range(len(fpred_dict[key])):
        ax[0, i].plot(t_dense - 3620., fpred_dense_dict[key][s], color=f'C{j-1}', alpha=0.1, zorder=-1)

    fpred_median = jnp.median(fpred_dict[key], axis=0)
    loo_i = az.loo(samples_mcmc_az.isel(chain=[j-1]), pointwise=True)['loo_i'].values

    vmin, vmax = np.percentile(loo_i, [5, 95.])
    im_loo = ax[1, i].scatter(t - 3620., fobs - fpred_median , marker='o', c=loo_i, cmap='binary_r', zorder=-1, vmin=-4.5, vmax=-1.32)

    pareto_k = az.loo(samples_mcmc_az.isel(chain=[j-1]), pointwise=True)['pareto_k'].values
    im_khat = ax[2, i].scatter(t - 3620., fobs - fpred_median , marker='o', c=pareto_k, cmap='binary', zorder=-1, vmin=-0.3, vmax=0.4)

tmin, tmax = 2800, 4400
for _a in ax.reshape(-1):
    _a.set_xlim(tmin- 3620 - 50, tmax - 3620 + 50)
    _a.grid(alpha=0.5)

# colorbars
cax = ax[1, 1].inset_axes([1.04, 0.1, 0.02, 0.8])
plt.colorbar(im_loo, cax=cax, label="$\ln p(f_i|f_{-i})$")

cax = ax[2, 1].inset_axes([1.04, 0.1, 0.02, 0.8])
plt.colorbar(im_khat, cax=cax, label="$\hat k$")

for _a in ax[1, :]:
    _a.set(ylim=(-15, 15))

for _a in ax[2, :]:
    _a.set(ylim=(-15, 15),  xlabel='Time [HJD - 3620 days]')

ax[0, 0].set_ylabel("Flux")
ax[1, 0].set_ylabel("Residuals coloured by\npointwise LOO")
ax[2, 0].set_ylabel("Residuals coloured by\n Pareto shape parameter")

for _a in ax[:, 1]:
    _a.set_yticklabels([])

ax[0, 0].set_title("Mode 1")
ax[0, 1].set_title("Mode 4")

for _a in ax.reshape(-1):
    _a.set_rasterization_zorder(0)
fig.savefig(paths.figures/"single_lens_lightcurve_fits.pdf", bbox_inches="tight")

# Plot pointwise loo diagnostics
loo_i1 = az.loo(samples_mcmc_az.isel(chain=[0]), pointwise=True)['loo_i'].values
loo_i2 = az.loo(samples_mcmc_az.isel(chain=[1]), pointwise=True)['loo_i'].values
loo_i3 = az.loo(samples_mcmc_az.isel(chain=[2]), pointwise=True)['loo_i'].values
loo_i4 = az.loo(samples_mcmc_az.isel(chain=[3]), pointwise=True)['loo_i'].values

fig, ax = plt.subplots(2,1,figsize=(12, 6), sharex=True)
ax[0].errorbar(t - 3620., fobs, ferr,  marker='o', color='k', linestyle='None', alpha=0.2) 
ax[0].set_ylabel("Flux")

ax[1].plot(t - 3620, loo_i2 - loo_i1, 'C1o', alpha=0.6, label='Mode 2', zorder=-1)
ax[1].plot(t - 3620, loo_i3 - loo_i1, 'C2o', alpha=0.6, label='Mode 3', zorder=-1)
ax[1].plot(t - 3620, loo_i4 - loo_i1, 'C3o', alpha=0.6, label='Mode 4', zorder=-1)
ax[1].set_ylabel("$\Delta\mathrm{elpd}_\mathrm{psis-loo}$")
ax[1].legend(fontsize=14)

ax[1].set_ylim(-2, 2)

for a in ax:
    a.grid(alpha=0.5)
    a.set_xlim(t[0] - 3620 - 50, t[-1] - 3620 + 50)
    a.set_xlim(-400, 400)
    a.set_zorder(-1)

fig.savefig(paths.figures/"pointwise_loo.pdf", bbox_inches="tight")

# Plot posterior trajectories
fig, ax = plt.subplots(figsize=(7, 8))

for i, key in enumerate(['mode1', 'mode2', 'mode3', 'mode4']):
    for s in range(len(fpred_dict[key])):
        ax.plot(w_dense_dict[key][s].real, w_dense_dict[key][s].imag, color=f'C{i}', alpha=0.2, zorder=-1)
        ax.plot(w_dense_dict[key][s].real[-1], w_dense_dict[key][s].imag[-1], color=f'C{i}', alpha=0.4, marker='<', zorder=-1)
#ax.set_aspect(1)
ax.set(xlabel="$w_\mathrm{east}$ [Einstein radii]", ylabel="$w_\mathrm{north}$ [Einstein radii]")
ax.axvline(0., color='k', alpha=0.2, ls='--')
ax.axhline(0., color='k', alpha=0.2, ls='--')
ax.set_title("Source trajectories")

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='', color='C0', label='Mode 1',ls='-'),
    Line2D([0], [0], marker='', color='C1', label='Mode 2',ls='-'),
    Line2D([0], [0], marker='', color='C2', label='Mode 3',ls='-'),
    Line2D([0], [0], marker='', color='C3', label='Mode 4',ls='-'),
]
ax.legend(handles=legend_elements, fontsize=14)
ax.set_rasterization_zorder(0)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
fig.savefig(paths.figures/"single_lens_trajectories.pdf", bbox_inches="tight")


# Plot posterior samples
fig, ax = plt.subplot_mosaic(
    """
    AB
    CD
    """,
    figsize=(9,8),
    gridspec_kw={'height_ratios': [1, 3], 'width_ratios': [3, 1], 'wspace': 0.07, 'hspace': 0.1}
)

ax['B'].axis('off')

for i in range(4):
    _samples = samples_mcmc_az.isel(chain=[i]).stack(sample=('chain', 'draw')).posterior

    ax['C'].scatter(
        jnp.exp(_samples['ln_tE'].values),
        _samples['piE'].values,
        color=f'C{i}', alpha=0.1, label=f'Mode {i + 1}',
        zorder=-1
    )
    ax['A'].hist(
        jnp.exp(_samples['ln_tE'].values), histtype='step', 
        lw=2., color=f'C{i}', density=True, bins=20, zorder=-1

    );
    ax['D'].hist(
        _samples['piE'].values, histtype='step', orientation=u'horizontal',
        lw=2., color=f'C{i}', density=True, bins=20, zorder=-1

    );

# Custom legend with scatter point 
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='C0', label='Mode 1',ls=''),
    Line2D([0], [0], marker='o', color='C1', label='Mode 2',ls=''),
    Line2D([0], [0], marker='o', color='C2', label='Mode 3',ls=''),
    Line2D([0], [0], marker='o', color='C3', label='Mode 4',ls=''),
]

ax['C'].legend(handles=legend_elements, fontsize=14)
ax['D'].set(xticklabels=[], yticklabels=[]);
ax['A'].set(xticklabels=[], yticklabels=[]);
ax['C'].set(xlabel=r'$t_E$ [days]', ylabel=r'$\pi_E$ ');


ax['C'].set(xlim=(80, 180), ylim=(0.15,0.42))
ax['A'].set_xlim(80, 180)
ax['D'].set(ylim=(0.15,0.42))

ax['C'].set_rasterization_zorder(0)
ax['C'].xaxis.set_minor_locator(AutoMinorLocator())
ax['C'].yaxis.set_minor_locator(AutoMinorLocator())
fig.savefig(paths.figures/"single_lens_samples.pdf", bbox_inches="tight")


# Re-weight NUTS samples 
compare_dict_mcmc = {f'mode_{i + 1}': az.loo(samples_mcmc_az.isel(chain=[i]), pointwise=True) for i in range(4)}
weights_stacking = az.compare(compare_dict_mcmc)['weight']
weights_stacking = weights_stacking.values

weights_bma = az.compare(compare_dict_mcmc, method='BB-pseudo-BMA')['weight']
weights_bma = weights_bma.values
samples_mcmc_reweighted = mixture_draws(samples_mcmc_az.posterior, weights_bma)
samples_mcmc_reweighted_stacking = mixture_draws(samples_mcmc_az.posterior, weights_stacking)

params_names = ['ln_t0', 'ln_tE', 'u0', 'piEE', 'piEN']
chains_equal_weighted = np.loadtxt(
    paths.data/"output/single_lens/ultranest/chains/equal_weighted_post.txt", skiprows=1
)
nsamples = samples_mcmc_az.posterior.isel(chain=[0])['ln_tE'].shape[-1]
idcs = np.random.randint(0, chains_equal_weighted.shape[0], nsamples)
chains_equal_weighted = chains_equal_weighted[idcs, :]
samples_ultranest = {k: chains_equal_weighted[:, i] for i, k in enumerate(params_names)}
samples_ultranest['piE'] = np.sqrt(samples_ultranest['piEE']**2 + samples_ultranest['piEN']**2)

# Print weights
print("Pseudo-BMA weights:")
print(az.compare(compare_dict_mcmc, method='pseudo-BMA')['weight'])
print("\n")

print("Pseudo-BMA+ weights:")
print(az.compare(compare_dict_mcmc, method='BB-pseudo-BMA')['weight'])
print("\n")

print("Stacking weights:")
print(az.compare(compare_dict_mcmc)['weight'])


# Plot re-weighted samples
fig, ax = plt.subplot_mosaic(
    """
    AB
    CD
    """,
    figsize=(9,8),
    gridspec_kw={'height_ratios': [1, 3], 'width_ratios': [3, 1], 'wspace': 0.07, 'hspace': 0.1}
)

ax['B'].axis('off')

nbins = 18

# BB-pseudo-BMA stacking
ax['C'].scatter(
    jnp.exp(samples_mcmc_reweighted['ln_tE']),
    samples_mcmc_reweighted['piE'],
    color=f'C0', alpha=0.3, zorder=-1
)
ax['A'].hist(
    jnp.exp(samples_mcmc_reweighted['ln_tE']), histtype='step',
    lw=2., color=f'C0', density=True, bins=nbins, zorder=-1

);
ax['D'].hist(
    samples_mcmc_reweighted['piE'], histtype='step', orientation=u'horizontal',
    lw=2., color=f'C0', density=True, bins=nbins, zorder=-1

);

# Stacking
ax['C'].scatter(
    jnp.exp(samples_mcmc_reweighted_stacking['ln_tE']),
    samples_mcmc_reweighted_stacking['piE'],
    color=f'C1', alpha=0.3, zorder=-1
)
ax['A'].hist(
    jnp.exp(samples_mcmc_reweighted_stacking['ln_tE']), histtype='step',
    lw=2., color=f'C1', density=True, bins=nbins, zorder=-1

);
ax['D'].hist(
    samples_mcmc_reweighted_stacking['piE'], histtype='step', orientation=u'horizontal',
    lw=2., color=f'C1', density=True, bins=nbins, zorder=-1

);


# Nested Sampling
ax['C'].scatter(
    jnp.exp(samples_ultranest['ln_tE']),
    samples_ultranest['piE'],
    color=f'k', alpha=0.2, zorder=-1
)
ax['A'].hist(
    jnp.exp(samples_ultranest['ln_tE']), histtype='step',
    lw=2., color=f'k', density=True, bins=nbins, zorder=-1

);
ax['D'].hist(
    samples_ultranest['piE'], histtype='step', orientation=u'horizontal',
    lw=2., color=f'k', density=True, bins=nbins , zorder=-1

);

ax['D'].set(xticklabels=[], yticklabels=[]);
ax['A'].set(xticklabels=[], yticklabels=[]);
ax['C'].set(xlabel=r'$t_E$ [days]', ylabel=r'$\pi_E$ ');

ax['C'].set(xlim=(80, 180), ylim=(0.15,0.42))
ax['A'].set_xlim(80, 180)
ax['D'].set(ylim=(0.15,0.42))

legend_elements = [
    Line2D([0], [0], marker='o', color='k', label='Bayesian posterior (NS)',ls=''),
    Line2D([0], [0], marker='o', color='C0', label='BB-pseudo-BMA+',ls=''),
    Line2D([0], [0], marker='o', color='C1', label='Stacking',ls=''),
]


ax['C'].legend(handles=legend_elements, fontsize=14)

ax['C'].set_rasterization_zorder(0)
ax['C'].xaxis.set_minor_locator(AutoMinorLocator())
ax['C'].yaxis.set_minor_locator(AutoMinorLocator())

for _a in ax.values():
    _a.set_rasterization_zorder(0)

fig.savefig(paths.figures/"single_lens_samples_reweighted.pdf", bbox_inches="tight")
