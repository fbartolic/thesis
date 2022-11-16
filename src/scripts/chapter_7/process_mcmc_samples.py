import paths
import numpy as np
import yaml
import starry
import astropy.units as u
import arviz as az

from utils import (
    load_filter, planck, 
    integrate_planck_over_filter, 
    inverse_integrate_planck_over_filter, 
    starry_intensity_to_bbtemp,
)

from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')

import arviz as az

np.random.seed(42)

starry.config.lazy = False
starry.config.quiet = True

def initialize_map(ydeg, nw, x):
    map = starry.Map(ydeg, nw=nw)
    map[1:, :, :] = x[1:, :] / x[0]
    map.amp = x[0]
    return map

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

# System parameters
planet = "hd209"
filter_name = "f444w"

# Load orbital and system parameters
with open(f"../../data/mapping_exo/system_parameters/{planet}/orbital_params_planet.yaml", "rb") as handle:
    params_p = yaml.safe_load(handle)
with open(f"../../data/mapping_exo/system_parameters/{planet}/orbital_params_star.yaml", "rb") as handle:
    params_s = yaml.safe_load(handle)
    
# Load filter
filt = load_filter("../../data/mapping_exo/filter_files", name=f"{filter_name}")
mask = filt[1] > 0.002

# Wavelength grid for starry map (should match filter range)
wavelength_grid = np.linspace(4.5 - 1.2, 4.5 + 1.2, 50)

# Save temperature maps for simulated maps
res = 200
projection="orthogonal"

xs_true = np.load(paths.data/"output/mapping_exo/T341_1bar_ylms/coefficients.npy")

for i in range(7):
    x_sim = xs_true[i]
    map = initialize_map(20, len(wavelength_grid), x_sim)
    map_sim_temp = starry_intensity_to_bbtemp(
        map.render(res=res, projection=projection), wavelength_grid
    )
    path = paths.data/f"output/mapping_exo/T341_1bar_mcmc/hd209/time_{i}/sim_map_temp.npy"
    if not path.exists():
        np.save(path, map_sim_temp)

# Save mean temeperature map and posterior sample map for each set of MCMC samples
for snr in [15, 50]:
    for ydeg_inf in np.arange(1, 7):
        for i in range(7):
            print("snr:", snr, "ydeg:", ydeg_inf, "time:", i)
            samples = az.from_netcdf(
                paths.data/f"output/mapping_exo/T341_1bar_mcmc/hd209/time_{i}/snr_{snr}/l_{ydeg_inf}/samples.nc"
            )

            # Save mean inferred temperature map and posterior sample maps
            x_mean = samples.posterior.w.mean(dim=['chain', 'draw'])
            map = starry.Map(ydeg_inf)
            map[1:, :] = x_mean[1:]/x_mean[0]
            map.amp = x_mean[0]
            map_inf_temp = inferred_intensity_to_bbtemp(
                map.render(res=res, projection=projection), filt, params_s, params_p
            )

            map_inf_temp_samples = []
            for s in range(9):
                x_sample = samples.posterior.w.isel(chain=0).isel(draw=s).values
                map = starry.Map(ydeg_inf)
                map[1:, :] = x_sample[1:]/x_sample[0]
                map.amp = x_sample[0]
                m = inferred_intensity_to_bbtemp(
                    map.render(res=80, projection=projection), filt, params_s, params_p
                )
                map_inf_temp_samples.append(m)

            path = paths.data/f"output/mapping_exo/T341_1bar_mcmc/hd209/time_{i}/snr_{snr}/l_{ydeg_inf}/map_inf_temp.npz"
            if not path.exists():
                np.savez(
                    path,
                    mean=map_inf_temp,
                    samples=np.stack(map_inf_temp_samples),
                )
