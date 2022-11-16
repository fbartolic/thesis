import paths
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import arviz as az 

param_names = ['ln_t0', 'ln_tE', 'ln_u0', 'ln_s', 'ln_q', 'alpha', 'ln_rho']

def get_samples_from_ns(path):
    chains_equal_weighted = np.loadtxt(path, skiprows=1)[-5000:, :]
#    ll_vals_pointwise = np.stack([ll_fn_vbb(s, pointwise=True) for s in chains_equal_weighted])
    samples_ns = {name: chains_equal_weighted[-5000:, i] for i, name in enumerate(param_names)}
    samples_ns_az = az.from_dict(samples_ns)
#    samples_ns_az.add_groups(
#        log_likelihood=xr.DataArray(ll_vals_pointwise[None, :, :], dims=['chain', 'draw', 'y_dim_0'])
#    )
    return samples_ns_az

samples_2k = get_samples_from_ns(paths.data/"output/binary_lens/nlive_2000/chains/equal_weighted_post.txt")
samples_5k = get_samples_from_ns(paths.data/"output/binary_lens/nlive_5000/chains/equal_weighted_post.txt")
samples_10k = get_samples_from_ns(paths.data/"output/binary_lens/nlive_10000/chains/equal_weighted_post.txt")

for samples in (samples_2k, samples_5k, samples_10k):
    samples.posterior['s'] = np.exp(samples.posterior['ln_s'])
    samples.posterior['q'] = np.exp(samples.posterior['ln_q'])
    samples.posterior['t0'] = np.exp(samples.posterior['ln_t0'])
    samples.posterior['u0'] = np.exp(samples.posterior['ln_u0'])
    samples.posterior['tE'] = np.exp(samples.posterior['ln_tE'])
    samples.posterior['rho'] = np.exp(samples.posterior['ln_rho'])

fig, ax = plt.subplots(3, 5, figsize=(16, 13), gridspec_kw={
    'hspace':0.6, 'wspace':0.6
})

alpha = 0.015
# s vs q
ax[0, 0].scatter(
    samples_2k.posterior['s'].values[0],
    samples_2k.posterior['q'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)


ax[1, 0].scatter(
    samples_5k.posterior['s'].values[0],
    samples_5k.posterior['q'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)

ax[2, 0].scatter(
    samples_10k.posterior['s'].values[0],
    samples_10k.posterior['q'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)


# s vs u0
ax[0, 1].scatter(
    samples_2k.posterior['s'].values[0],
    samples_2k.posterior['u0'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)


ax[1, 1].scatter(
    samples_5k.posterior['s'].values[0],
    samples_5k.posterior['u0'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)

ax[2, 1].scatter(
    samples_10k.posterior['s'].values[0],
    samples_10k.posterior['u0'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)

#  t0 vs tE
ax[0, 2].scatter(
    samples_2k.posterior['t0'].values[0],
    samples_2k.posterior['tE'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)

ax[1, 2].scatter(
    samples_5k.posterior['t0'].values[0],
    samples_5k.posterior['tE'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)

ax[2, 2].scatter(
    samples_10k.posterior['t0'].values[0],
    samples_10k.posterior['tE'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)

# tE vs u0
ax[0, 3].scatter(
    samples_2k.posterior['tE'].values[0],
    samples_2k.posterior['u0'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)


ax[1, 3].scatter(
    samples_5k.posterior['tE'].values[0],
    samples_5k.posterior['u0'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)

ax[2, 3].scatter(
    samples_10k.posterior['tE'].values[0],
    samples_10k.posterior['u0'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)

# alpha vs s
ax[0, 4].scatter(
    np.rad2deg(samples_2k.posterior['alpha'].values[0]),
    samples_2k.posterior['s'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)


ax[1, 4].scatter(
    np.rad2deg(samples_5k.posterior['alpha'].values[0]),
    samples_5k.posterior['s'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)

ax[2, 4].scatter(
    np.rad2deg(samples_10k.posterior['alpha'].values[0]),
    samples_10k.posterior['s'].values[0],
    marker='o',
    color='k',
    alpha=alpha,
    zorder=-1,
)


for i in range(3):
    ax[i, 0].set(xlim=(1.56831219, 1.91554083), ylim=(0.18451952, 0.2780373))
    ax[i, 1].set(ylim=(-0.02, 0.201), xlim=(1.56831219, 1.91554083))
    ax[i, 2].set(xlim=(7831, 7836), ylim=(140, 240))
    ax[i, 3].set(xlim=(140, 240), ylim=(-0.02, 0.201))
    ax[i, 4].set(ylim=(1.568, 1.915), xlim=(-90, 90))

    ax[i, 0].set(xlabel=r'$s$', ylabel=r'$q$')
    ax[i, 1].set(xlabel=r'$s$', ylabel=r'$u_0$')
    ax[i, 2].set(xlabel=r'$t_0$', ylabel=r'$t_E$')
    ax[i, 3].set(xlabel=r'$tE$', ylabel=r'$u_0$')
    ax[i, 4].set(xlabel=r'$\alpha$', ylabel=r'$s$')



for a in ax.reshape(-1):
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())
    a.set_rasterization_zorder(0)

ax[0, 2].set_title("$n_\mathrm{live}=2000$", pad=30, fontsize=25, fontweight='bold')
ax[1, 2].set_title("$n_\mathrm{live}=5000$", pad=30, fontsize=25, fontweight='bold')
ax[2, 2].set_title("$n_\mathrm{live}=10000$", pad=30, fontsize=25, fontweight='bold')

fig.savefig(paths.figures/"ns_pairplot.pdf", bbox_inches='tight')