import paths
import yaml
import numpy as np
import starry
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
import astropy.units as u
import arviz as az

from utils import (
    starry_intensity_to_bbtemp,
    load_filter,
    integrate_planck_over_filter,
    inverse_integrate_planck_over_filter
)

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


def get_preimage(A, fsim, ydeg_inf=20):
    ferr = 1e-6 * np.random.rand(len(fsim))
    fobs = fsim + np.random.normal(0, ferr[0], size=len(fsim))

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
    x_lsq, _ = starry.linalg.solve(A_full, fobs, C=1e-03** 2, L=L)
    return x_lsq[4:]

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

texp = 5.5*u.s

# Wavelength grid for starry map (should match filter range)
wavelength_grid = np.linspace(4.5 - 1.2, 4.5 + 1.2, 50)

# Load data
fsim_list = np.load(paths.data/"output/mapping_exo/T341_1bar_fluxes/hd209/fsim_snapshots.npy")
fsim_ref_list = np.load(paths.data/"output/mapping_exo/T341_1bar_fluxes/hd209/fsim_reference_snapshots.npy")
t_eclipse = np.load(paths.data/"output/mapping_exo/T341_1bar_fluxes/hd209/t_eclipse.npy")

# Normalize
norm = np.max(fsim_ref_list, axis=1)
fsim_ref_list /= norm[:, None]
fsim_list /= norm[:, None]

# Simulated temperature maps
xs_sim = np.load(paths.data/"output/mapping_exo/T341_1bar_ylms/coefficients.npy")

maps_sim_temp = []
for i in range(7):
    x_sim = xs_sim[i]
    map = initialize_map(20, len(wavelength_grid), x_sim)
    map_sim_temp = starry_intensity_to_bbtemp(
        map.render(res=200, projection="Orthogonal"), wavelength_grid
    ) 
    maps_sim_temp.append(map_sim_temp)

# Simulated temperature maps (preimages)
A = get_design_matrix(t_eclipse, 20, params_s, params_p, texp)
maps_sim_temp_preimages = []

for i in range(7):
    x_preimage = get_preimage(A, fsim_list[i])
    map = starry.Map(20)
    map[1:, :] = x_preimage[1:]/x_preimage[0]
    map.amp = x_preimage[0]

    map_inf_temp = inferred_intensity_to_bbtemp(
        map.render(res=200, projection="Orthogonal"), filt, params_s, params_p
    )
    maps_sim_temp_preimages.append(map_inf_temp)


# Inferred temperature maps
l = 4
maps_inf_temp1 = []
for i in range(7):
    dat = np.load(
        paths.data/f"output/mapping_exo/T341_1bar_mcmc/hd209/time_{i}/snr_15/l_{l}/map_inf_temp.npz"
    ) 
    maps_inf_temp1.append(dat["mean"])

maps_inf_temp2 = []
for i in range(7):
    dat = np.load(
        paths.data/f"output/mapping_exo/T341_1bar_mcmc/hd209/time_{i}/snr_50/l_{l}/map_inf_temp.npz"
    ) 
    maps_inf_temp2.append(dat["mean"])

# Figure
fig, ax = plt.subplots(4, 7, figsize=(12, 8),  gridspec_kw={'hspace':0.4})

vmin, vmax= 1400, 1700
norm = colors.Normalize(vmin=vmin, vmax=vmax)

for i in range(len(maps_sim_temp)):
    map.show(
        image=maps_sim_temp[i], projection="orthogonal", cmap="turbo", 
        norm=norm,
        ax=ax[0, i],
        zorder=-1,
    )
    map.show(
        image=maps_sim_temp_preimages[i], projection="orthogonal", cmap="turbo", 
        norm=norm,
        ax=ax[1, i],
        zorder=-1,
    )

    map.show(
        image=maps_inf_temp1[i], projection="orthogonal", cmap="turbo", 
        norm=norm,
        ax=ax[2, i],
        zorder=-1,
    )
    ax[0, i].set_title(f"$t/P = {i}$")

    map.show(
        image=maps_inf_temp2[i], projection="orthogonal", cmap="turbo", 
        norm=norm,
        ax=ax[3, i],
        zorder=-1,
    )
 

for a in ax.reshape(-1):
    a.set_aspect(1)
    a.set_rasterization_zorder(0)

ax[0, 3].set_title("Simulated maps\n\nt/P=3")
ax[1, 3].set_title("Simulated maps (preimages)", pad=15)
ax[2, 3].set_title("Inferred (mean) maps, $l=4$, SNR=15", pad=15)
ax[3, 3].set_title("Inferred (mean) maps, $l=4$, SNR=50", pad=15)

# Colorbar
cax = fig.add_axes([0.36, 0.02, 0.3, 0.03])
fig.colorbar(
mpl.cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap="turbo"),
    cax=cax,
        orientation='horizontal',
        label="blackbody temperature [K]", fraction=1.
)

fig.savefig(paths.figures/"simulated_vs_inferred.pdf", bbox_inches="tight")

# Plot flux residuals
fig, ax = plt.subplots(2, 7, figsize=(13, 5), sharey=True, 
gridspec_kw={'wspace':0.1, 'hspace':0.5})
l = 4

for j, snr in enumerate([15, 50]):
    for i in range(7):
        # Plot flux
        samples = az.from_netcdf(
            paths.data/f"output/mapping_exo/T341_1bar_mcmc/hd209/time_{i}/snr_{snr}/l_{l}/samples.nc"
        )
        fsim = np.load(paths.data/"output/mapping_exo/T341_1bar_fluxes/hd209/fsim_snapshots.npy")[i]
        fsim_ref = np.load(paths.data/"output/mapping_exo/T341_1bar_fluxes/hd209/fsim_reference_snapshots.npy")[i]

        # Normalize
        norm = np.max(fsim_ref)
        fsim_ref /= norm
        fsim /= norm

        fpred_samples = samples.posterior.isel(chain=0).fpred.values

#        ax[j, i].fill_between(
#            t_eclipse/3.5,
#            (np.percentile(fpred_samples, 16, axis=0) - fsim_ref)*1e06,
#            (np.percentile(fpred_samples, 84, axis=0) - fsim_ref)*1e06,
#            color="C0",
#            alpha=0.5,
#        )

        for s in np.random.randint(0, fpred_samples.shape[0], size=50):
            ax[j, i].plot(t_eclipse/3.5, (fpred_samples[s] - fsim_ref)*1e06, alpha=0.1, color='C0')
        ax[j, i].plot(t_eclipse/3.5 ,(fsim - fsim_ref)*1e06, 'k', alpha=0.6)

for a in ax.reshape(-1):
    a.set(xlim=(-0.025, 0.025), ylim=(-20, 20))
#    a.xaxis.set_minor_locator(AutoMinorLocator())
#    a.yaxis.set_minor_locator(AutoMinorLocator())
    a.set_xticks([-0.015,  0.015])
    a.xaxis.set_major_formatter('{x:.2f}')


for i, a in enumerate(ax[0, :]):
    a.set_xticklabels([])
    a.set_title(f"$t/P = {i}$")

ax[0, 3].set_title(f"SNR = 15\n$t/P = 3$")
ax[1, 3].set_title("SNR = 50", pad=20)

for a in ax[:, 0]:
    a.set_ylabel("flux difference\n[ppm]")

fig.savefig(paths.figures/"simulated_vs_inferred_flux.pdf", bbox_inches="tight")
