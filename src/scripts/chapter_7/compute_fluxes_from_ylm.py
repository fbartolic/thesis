import paths
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

import starry
import astropy.units as u
import pickle as pkl
import yaml

import os
import sys

from utils import (
    load_filter,
    planck,
    starry_intensity_to_bbtemp,
    integrate_planck_over_filter,
    inverse_integrate_planck_over_filter,
)


np.random.seed(42)
starry.config.lazy = False
starry.config.quiet = True


def compute_simulated_flux(
    t, map_star, map_planet, params_s, params_p, filt, wavelength_grid, texp,
):
    # Interpolate filter throughput
    thr_interp = np.interp(wavelength_grid, filt[0], filt[1])

    # Ratio of star and planet map *ampliudes* needs to be proportional to
    # (Rp/Rs)**2 so we need to multiply the planet map amplitude with that factor
    radius_ratio = params_p["r"] * u.Rjupiter.to(u.Rsun) / params_s["r"]
    map_planet.amp *= radius_ratio ** 2

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

    # Compute flux
    A = sys.design_matrix(t)
    x = np.concatenate([map_star.amp * map_star._y, map_planet.amp * map_planet._y])
    fsim_spectral = np.tensordot(A, x, axes=1)

    wav_filt = filt[0]
    throughput = filt[1]

    # Interpolate filter throughput to map wavelength grid
    throughput_interp = np.interp(wavelength_grid, wav_filt, throughput)

    # Integrate flux over bandpass
    fsim = np.trapz(
        fsim_spectral * throughput_interp, axis=1, x=wavelength_grid * u.um.to(u.m)
    )

    # Rescale the amplitude of the planet map back to its original value
    map_planet.amp *= radius_ratio ** (-2.0)

    return fsim, sys


def initialize_featureless_map(T_star, wavelength_grid, ydeg=1):
    # Initialize star map
    map_star = starry.Map(ydeg=1, nw=len(wavelength_grid))
    Llam = (4 * np.pi) * np.pi * planck(T_star, wavelength_grid).value
    map_star.amp = Llam / 4
    return map_star


def get_lower_order_map(map, ydeg=2):
    assert map.ydeg > ydeg
    x = map._y * map.amp
    x = x[: (ydeg + 1) ** 2]
    map = starry.Map(ydeg, nw=map.nw)
    map[1:, :, :] = x[1:, :] / x[0]
    map.amp = x[0]

    return map


def draw_sample_lightcurve(t, fsim, sigma=None, snr=None):
    eclipse_depth = np.max(fsim) - np.min(fsim)
    sigma = eclipse_depth / snr
    fobs = fsim + np.random.normal(0, sigma, size=len(t))
    return fobs, sigma * np.ones_like(fobs)


def inferred_intensity_to_bbtemp(I_planet_raw, filt, params_s, params_p):
    """
    Convert inferred starry intensity map to a BB temperature map.
    """
    # Star spectral radiance integrated over solid angle and bandpass
    I_star = np.pi * integrate_planck_over_filter(params_s["Teff"], filt)

    # Rescale the intensity of the planet map to physical units
    I_planet = (
        I_planet_raw
        * I_star
        * (params_s["r"] / (params_p["r"] * u.Rjupiter.to(u.Rsun))) ** 2
    )

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


# Load Ylm coefficients for each snapshot
input_dir = paths.data/"output/mapping_exo/T341_1bar_ylms/"
xs = np.load(os.path.join(input_dir, "coefficients.npy"))
t_snapshots = np.load(os.path.join(input_dir, "times.npy"))

# System parameters
planet = "hd209"
filter_name = "f444w"

# Load orbital and system parameters
with open(
    paths.data/f"mapping_exo/system_parameters/{planet}/orbital_params_planet.yaml", "rb"
) as handle:
    params_p = yaml.safe_load(handle)
with open(
    paths.data/f"mapping_exo/system_parameters/{planet}/orbital_params_star.yaml", "rb"
) as handle:
    params_s = yaml.safe_load(handle)

# Load filter
filt = load_filter(paths.data/"mapping_exo/filter_files", name=f"{filter_name}")
mask = filt[1] > 0.002

# Wavelength grid for starry map (should match filter range)
wavelength_grid = np.linspace(4.5 - 1.2, 4.5 + 1.2, 50)

# Set exposure time
texp = 5.449 * u.s

# Generate observation times excluding transit
porb = params_p["porb"] * u.d
t0 = 0.5 * params_p["porb"] * u.d
t_ = np.linspace(-t0.value, +t0.value, int(porb.to(u.s) / texp))

## Mask transit to obtain everything but the transit
# mask_tran = np.abs(t_) > 0.9
# t_complete = t_[~mask_tran]

# Only eclipse
# delta_t = 0.1 * porb
delta_t = 0.25
mask_eclipse = np.logical_and(t_ > -delta_t, t_ < delta_t)
t_eclipse = t_[mask_eclipse]

# Load simulation snapshots as starry maps
ydeg = 20
map_star = initialize_featureless_map(params_s["Teff"], wavelength_grid)

fsim_list = []
fsim_reference_list = []

# Select one week of data
xs = xs[:7]

for i, x in enumerate(xs):
    print(i)
    # Initialize planet map
    map_snapshot = initialize_map(ydeg, len(wavelength_grid), x)
    map_snapshot_quadrupole = get_lower_order_map(map_snapshot, ydeg=2)

    # Simulate reference light curves
    fsim_reference, _ = compute_simulated_flux(
        t_eclipse,
        map_star,
        map_snapshot_quadrupole,
        params_s,
        params_p,
        filt,
        wavelength_grid,
        texp,
    )
    fsim, sys = compute_simulated_flux(
        t_eclipse,
        map_star,
        map_snapshot,
        params_s,
        params_p,
        filt,
        wavelength_grid,
        texp,
    )

    fsim_list.append(fsim)
    fsim_reference_list.append(fsim_reference)

fsim_snapshots = np.stack(fsim_list)
fsim_reference_snapshots = np.stack(fsim_reference_list)

# Save to file
output_dir = paths.data/"output/mapping_exo/T341_1bar_fluxes/hd209/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
np.save(os.path.join(output_dir, "fsim_snapshots.npy"), fsim_snapshots)
np.save(
    os.path.join(output_dir, "fsim_reference_snapshots.npy"), fsim_reference_snapshots
)
np.save(os.path.join(output_dir, "t_eclipse.npy"), t_eclipse)
