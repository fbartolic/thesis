import paths
import numpy as np
import yaml
import starry
import astropy.units as u
import os
import arviz as az

from utils import (
    load_filter, planck, 
    integrate_planck_over_filter, 
    inverse_integrate_planck_over_filter, 
)

import jax.numpy as jnp
from jax import random
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import numpyro
import numpyro.distributions as dist
from numpyro.infer import *
numpyro.set_host_device_count(4)

import arviz as az

np.random.seed(42)

starry.config.lazy = False
starry.config.quiet = True


def initialize_featureless_map(T_star, wavelength_grid, ydeg=1):
    # Initialize star map
    map_star = starry.Map(ydeg=1, nw=len(wavelength_grid))
    Llam = (4 * np.pi) * np.pi * planck(T_star, wavelength_grid).value
    map_star.amp = Llam / 4
    return map_star


def get_lower_order_map(map, ydeg=2):
    assert map.ydeg > ydeg
    x = map._y*map.amp
    x = x[:(ydeg + 1)**2]
    map = starry.Map(ydeg, nw=map.nw)
    map[1:, :, :] = x[1:, :]/x[0]
    map.amp = x[0]
    
    return map

def draw_sample_lightcurve(t, fsim, sigma=None, snr=None):
    eclipse_depth = np.max(fsim) - np.min(fsim)
    sigma = eclipse_depth / snr
    fobs = fsim + np.random.normal(0, sigma, size=len(t))
    return fobs, sigma*np.ones_like(fobs)


def inferred_intensity_to_bbtemp(I_planet_raw, filt, params_s, params_p):
    """
    Convert inferred starry intensity map to a BB temperature map.
    """
    # Star spectral radiance integrated over solid angle and bandpass
    I_star = np.pi * integrate_planck_over_filter(params_s["Teff"], filt)

    # Rescale the intensity of the planet map to physical units
    I_planet = I_planet_raw * I_star * (params_s["r"] / (params_p["r"]*u.Rjupiter.to(u.Rsun))) ** 2

    # Plot temperature map of the planet
    bbtemp_map_inf = np.copy(I_planet[:, :].value)

    for i in range(I_planet.shape[0]):
        for j in range(I_planet.shape[1]):
            bbtemp_map_inf[i, j] = inverse_integrate_planck_over_filter(
                I_planet[i, j].value, filt
            )
    return bbtemp_map_inf

def initialize_map(ydeg, nw, x):
    map = starry.Map(ydeg, nw=nw)
    map[1:, :, :] = x[1:, :] / x[0]
    map.amp = x[0]
    return map

def get_model(
    A,
    fobs_trial,
    ferr_trial,
    ydeg_inf=5,
    sigmoid_constraint=True,
    sigmoid_steepness=1e6,
    sd=1e-02,
):
    ncoeff = (ydeg_inf + 1) ** 2

    # First we solve the least squares problem to get an estimate of stellar flux fs
    # This parameter is problematic for the sampler for some reason so we fit for a
    # small deviation from the least squares estimate
    ncoeff_prim = 4  # stellar map is l=1
    A_full = A[:, : ncoeff_prim + (ydeg_inf + 1) ** 2]
    A_sec = A_full[:, ncoeff_prim:]

    # Primary
    L_prim = np.ones(ncoeff_prim)
    L_prim[1:] = 1e-10 ** 2
    L_sec = 1e-02 ** 2 * np.ones((ydeg_inf + 1) ** 2)
    L_sec[1:] = 1e-03 ** 2
    L = np.concatenate([L_prim, L_sec])

    x_lsq, _ = starry.linalg.solve(A_full, fobs_trial, C=ferr_trial ** 2, L=L,)
    fs_lsq = x_lsq[0]

    # Transform the Ylm coefficients to simplify the structure of the covariance matrix
    # Polar transform
    R = starry.Map(ydeg_inf).ops.dotR(
        np.eye(ncoeff),
        np.array(1.0),
        np.array(0.0),
        np.array(0.0),
        np.array(-np.pi / 2),
    )

    # Transform to group Ylms by order
    idx = [[] for m in range(2 * ydeg_inf + 1)]
    for m in range(ydeg_inf + 1):
        l = np.arange(abs(m), ydeg_inf + 1)
        idx[2 * m] = l ** 2 + l + m
        if m > 0:
            idx[2 * m - 1] = l ** 2 + l - m
    Nb = np.array([len(i) for i in idx], dtype=int)
    ii = np.array([item for sublist in idx for item in sublist], dtype=int)
    G = np.zeros((ncoeff, ncoeff))
    G[np.arange(ncoeff), ii] = 1

    # Full transform
    Q = G @ R
    QInv = np.linalg.inv(Q)

    map = starry.Map(ydeg_inf)
    _, _, Y2P, _, _, _ = map.get_pixel_transforms(oversample=4)

    def model(fobs, ferr):
        u = []
        j = 0
        for i, nb in enumerate(Nb):
            u.append(
                numpyro.sample(f"u_{i}", dist.Normal(jnp.zeros(nb), sd * jnp.ones(nb)))
            )
            j += nb
        u = jnp.concatenate(u)
        w = numpyro.deterministic("w", jnp.dot(jnp.array(QInv), u))

        if sigmoid_constraint:
            # Penalize values of `p` outside [0, 1]
            p = numpyro.deterministic("p", jnp.dot(jnp.array(Y2P), w)).reshape(-1)
            penalty = -jnp.log(1.0 + jnp.exp(-sigmoid_steepness * p.reshape(-1)))
            # s = 1e3
            # penalty = (
            #    s * p
            #    - jnp.log(jnp.exp(s * p) + 1)
            #    + s * (1 - p)
            #    - jnp.log(jnp.exp(s * (1 - p)) + 1)
            # )
            numpyro.factor("pot", penalty.sum())

        fp = jnp.dot(A_sec, w).reshape(-1)

        fs_delta = numpyro.sample("fs_delta", dist.Normal(0., 1e-03), sample_shape=(1,))
        f = (fs_lsq + fs_delta) + fp
        numpyro.deterministic("fpred", f)

        numpyro.sample(
            "obs", dist.Normal(jnp.array(f), jnp.array(ferr)), obs=jnp.array(fobs),
        )

    u_start = 1e-04 * np.random.randn(ncoeff)
    init_vals = {"fs_delta": 0.0, "sd": 1e-04}
    j = 0
    for i, nb in enumerate(Nb):
        init_vals[f"u_{i}"] = u_start[j : j + nb]
        j += nb

    return model, init_vals, Nb

def get_design_matrix(t, ydeg, params_s, params_p, texp):
    map_planet = starry.Map(ydeg)
    map_star = starry.Map(1)

    # Initialize system
    star = starry.Primary(map_star, r=params_s["r"] * u.Rsun, m=params_s["m"] * u.Msun)
    planet = starry.Secondary(
        map_planet,
        r=params_p["r"] * (u.Rjupiter.to(u.Rsun)) * u.Rsun,
        porb=params_p["porb"] * u.d,
        prot=params_p["prot"] * u.d,
        t0=0.5 * params_p["porb"] * u.d,
        inc=params_p["inc"] * u.deg,
        theta0=180,
    )

    sys = starry.System(star, planet, texp=(texp.to(u.d)).value, oversample=9, order=0)
    return sys.design_matrix(t)

# System parameters
planet = "hd209"
filter_name = "f444w"

# Load orbital and system parameters
with open(paths.data/f"mapping_exo/system_parameters/{planet}/orbital_params_planet.yaml", "rb") as handle:
    params_p = yaml.safe_load(handle)
with open(paths.data/f"mapping_exo/system_parameters/{planet}/orbital_params_star.yaml", "rb") as handle:
    params_s = yaml.safe_load(handle)
    
# Load filter
filt = load_filter(paths.data/"mapping_exo/filter_files", name=f"{filter_name}")
mask = filt[1] > 0.002

# Wavelength grid for starry map (should match filter range)
wavelength_grid = np.linspace(4.5 - 1.2, 4.5 + 1.2, 50)

# Set exposure time
texp = 5.449*u.s

input_dir = paths.data/"output/mapping_exo/T341_1bar_fluxes/hd209/"
t_snapshots = np.load(os.path.join(input_dir, "fsim_snapshots.npy"))
xs = np.load(os.path.join(input_dir, "coefficients.npy"))
fsim_list = np.load(os.path.join(input_dir, "fsim_snapshots.npy"))
fsim_ref_list = np.load(os.path.join(input_dir, "fsim_reference_snapshots.npy"))
t_eclipse = np.load(os.path.join(input_dir, "t_eclipse.npy"))

# Normalize
norm = np.max(fsim_ref_list, axis=1)
fsim_ref_list /= norm[:, None]
fsim_list /= norm[:, None]

ydeg_list = np.arange(1, 7)

# Signal to noise ratio on the secondary eclipse depth
#for snr in [15, 50]:
for snr in [15, 50]:
    # Generate mock light curves
    fobs_trial, ferr = draw_sample_lightcurve(t_eclipse, fsim_list[0], snr=snr)
    lc_list = []
    for fsim in fsim_list:
        fobs, ferr = draw_sample_lightcurve(t_eclipse, fsim, snr=snr)
        lc = np.stack([fobs, ferr]) 
        lc_list.append(lc)
    lc_list = np.stack(lc_list)

    # Fit maps with different degrees
#    for ydeg_inf in ydeg_list:
    for ydeg_inf in [12]:
        samples_list = []
        A = get_design_matrix(t_eclipse, ydeg_inf, params_s, params_p, texp)
        model, init_vals, Nb = get_model(
            A, fobs_trial, ferr, ydeg_inf=ydeg_inf, sd=5e-05, sigmoid_constraint=True, 
            sigmoid_steepness=1e4
        )
        loglike_fn = lambda p: numpyro.infer.util.log_likelihood(model, p, fobs_trial, ferr)
        ncoeff = (ydeg_inf + 1)**2

        nuts_kernel = NUTS(
            model,
            dense_mass=[(f"u_{i}",) for i in Nb],
            init_strategy=init_to_value(values=init_vals),
            target_accept_prob=0.9,
            max_tree_depth=10,
        )
        ntune = 500
        ndraws = 1000
        nchains = 2

        mcmc = MCMC(
            nuts_kernel,
            num_warmup=ntune,
            num_samples=ndraws,
            num_chains=nchains,
            progress_bar=False,
        )
        key = random.PRNGKey(0)

        # Iterate over all light curves
        for i, lc in enumerate(lc_list):
            fobs, ferr = lc
            save_dir = paths.data/f"output/mapping_exo/T341_1bar_mcmc/hd209/time_{i}/snr_{snr}/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(
                os.path.join(save_dir, "lightcurve.npy"), lc_list[i]
            )

            key, subkey = random.split(key)
            mcmc.run(subkey, fobs, ferr)
            samples = mcmc.get_samples()
            samples_az = az.from_numpyro(mcmc)
            save_dir = paths.data/f"output/mapping_exo/T341_1bar_mcmc/hd209/time_{i}/snr_{snr}/l_{ydeg_inf}/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            az.to_netcdf(
                samples_az, 
                os.path.join(save_dir, f"samples.nc")
            )
        print(f"Finished ydeg={ydeg_inf}")
