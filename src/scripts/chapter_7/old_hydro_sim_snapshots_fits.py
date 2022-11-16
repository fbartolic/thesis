import numpy as np
import yaml
import starry
import astropy.units as u

from matplotlib import colors
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import (
    load_filter,
    planck,
    starry_intensity_to_bbtemp,
    simulation_snapshot_to_ylm,
    integrate_planck_over_filter,
    inverse_integrate_planck_over_filter,
    fit_model_num,
)

np.random.seed(42)

starry.config.lazy = False
starry.config.quiet = True


def simulation_snapshot_to_ylm(path, wavelength_grid, ydeg=20, temp_offset=-450):
    data = np.loadtxt(path)

    nlat = 512
    nlon = 1024

    lons = np.linspace(-180, 180, nlon)
    lats = np.linspace(-90, 90, nlat)

    lon_grid, grid = np.meshgrid(lons, lats)

    temp_grid = np.zeros_like(lon_grid)

    for i in range(nlat):
        for j in range(nlon):
            temp_grid[i, j] = data.reshape((nlat, nlon))[i, j]

    temp_grid = np.roll(temp_grid, int(temp_grid.shape[1] / 2), axis=1) + temp_offset

    x_list = []
    map_tmp = starry.Map(ydeg)

    # Evaluate at fewer points for performance reasons
    idcs = np.linspace(0, len(wavelength_grid) - 1, 10).astype(int)
    for i in idcs:
        I_grid = np.pi * planck(temp_grid, wavelength_grid[i])
        map_tmp.load(I_grid, force_psd=True)
        x_list.append(map_tmp._y * map_tmp.amp)

    # Interpolate to full grid
    x_ = np.vstack(x_list).T
    x_interp_list = [
        np.interp(wavelength_grid, wavelength_grid[idcs], x_[i, :])
        for i in range(x_.shape[0])
    ]
    x = np.vstack(x_interp_list)

    return x


def compute_simulated_lightcurve(
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
    wav_filt = filt[0]
    throughput = filt[1]

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


# System parameters
planet = "hd189"
filter_name = "f444w"

# Load orbital and system parameters
with open(
    f"../../data/system_parameters/{planet}/orbital_params_planet.yaml", "rb"
) as handle:
    params_p = yaml.safe_load(handle)
with open(
    f"../../data/system_parameters/{planet}/orbital_params_star.yaml", "rb"
) as handle:
    params_s = yaml.safe_load(handle)

# Load filter
filt = load_filter(name=f"{filter_name}")
mask = filt[1] > 0.002

# Wavelength grid for starry map (should match filter range)
wavelength_grid = np.linspace(filt[0][mask][0], filt[0][mask][-1], 100)

# Set exposure time
texp = 3.746710 * u.s

# Signal to noise ratio on the secondary eclipse depth
snr_ratios = [20, 100]

# Load simulation snapshots as starry maps
ydeg = 25

snapshots_ylm = [
    simulation_snapshot_to_ylm(
        f"../../data/hydro_snapshots_raw/T341_temp_{day}days.txt",
        wavelength_grid,
        ydeg=ydeg,
        temp_offset=-450,
    )
    for day in [100, 106, 108, 109]
]


def initialize_map(ydeg, nw, x):
    map = starry.Map(ydeg, nw=nw)
    map[1:, :, :] = x[1:, :] / x[0]
    map.amp = x[0]
    return map


snapshots_maps = [initialize_map(ydeg, len(wavelength_grid), x) for x in snapshots_ylm]
snapshots_maps_quadrupole = [get_lower_order_map(map, ydeg=2) for map in snapshots_maps]


# Generate observation times excluding transit
porb = params_p["porb"] * u.d
t0 = 0.5 * params_p["porb"] * u.d

t_ = np.linspace(-t0.value, +t0.value, int(0.5 * porb.to(u.s) / texp))

# Mask transit
mask_tran = np.abs(t_) > 0.9
t_complete = t_[~mask_tran]
t_eclipse = np.linspace(-0.1, +0.1, int(0.5 * porb.to(u.s) / texp))


def initialize_figure():
    fig = plt.figure(figsize=(16, 10))

    gs = fig.add_gridspec(
        nrows=7,
        ncols=4 + 1 + 4 + 1 + 4 + 1 + 4,
        height_ratios=[3, 2.5, 3, 2, 2.5, 3, 2],
        width_ratios=4 * [1] + [0.4] + 4 * [1] + [0.4] + 4 * [1] + [0.4] + 4 * [1],
        hspace=0.0,
        wspace=0.02,
    )

    # Axes for the simulated maps
    ax_sim_maps = [
        fig.add_subplot(gs[0, :4]),
        fig.add_subplot(gs[0, 5:9]),
        fig.add_subplot(gs[0, 10:14]),
        fig.add_subplot(gs[0, 15:19]),
    ]

    ax_text = [
        fig.add_subplot(gs[1, :]),
        fig.add_subplot(gs[4, :]),
    ]

    # Axes for inferred maps
    ax_inf_maps = {snr: [] for snr in snr_ratios}
    ax_samples = {snr: [] for snr in snr_ratios}

    for idx, snr in zip([2, 5], snr_ratios):
        ax_inf_maps[snr] = [
            fig.add_subplot(gs[idx, :4]),
            fig.add_subplot(gs[idx, 5:9]),
            fig.add_subplot(gs[idx, 10:14]),
            fig.add_subplot(gs[idx, 15:19]),
        ]

        # Axes for samples
        ax_s1 = [fig.add_subplot(gs[idx + 1, i]) for i in range(4)]
        ax_s2 = [fig.add_subplot(gs[idx + 1, 5 + i]) for i in range(4)]
        ax_s3 = [fig.add_subplot(gs[idx + 1, 10 + i]) for i in range(4)]
        ax_s4 = [fig.add_subplot(gs[idx + 1, 15 + i]) for i in range(4)]
        ax_samples[snr] = [ax_s1, ax_s2, ax_s3, ax_s4]

    return fig, ax_sim_maps, ax_text, ax_inf_maps, ax_samples


def main(t):
    map_star = initialize_featureless_map(params_s["Teff"], wavelength_grid)

    # Simulate reference light curves
    fsim_reference_list = []
    fsim_list = []

    fobs_list = {snr: [] for snr in snr_ratios}
    ferr_list = {snr: [] for snr in snr_ratios}

    for i in range(len(snapshots_maps_quadrupole)):
        fsim_reference, _ = compute_simulated_lightcurve(
            t,
            map_star,
            snapshots_maps_quadrupole[i],
            params_s,
            params_p,
            filt,
            wavelength_grid,
            texp,
        )
        fsim, sys = compute_simulated_lightcurve(
            t,
            map_star,
            snapshots_maps[i],
            params_s,
            params_p,
            filt,
            wavelength_grid,
            texp,
        )

        lc_norm = np.max(fsim_reference)

        fsim_reference = fsim_reference / lc_norm
        fsim = fsim / lc_norm

        fsim_reference_list.append(fsim_reference)
        fsim_list.append(fsim)

        for snr in snr_ratios:
            fobs, ferr = draw_sample_lightcurve(t, fsim, snr=snr)
            fobs_list[snr].append(fobs)
            ferr_list[snr].append(ferr)

    # Compute the design matrix
    ydeg_inf = 5
    A = sys.design_matrix(t)

    # Solve for the map coefficients
    traces = {snr: [] for snr in snr_ratios}
    traces_az = {snr: [] for snr in snr_ratios}

    tune = 1000
    draws = 500
    for i in range(4):
        for snr in snr_ratios:
            trace, trace_az = fit_model_num(
                A,
                fobs_list[snr][i],
                ferr_list[snr][i],
                tune=tune,
                draws=draws,
                chains=2,
            )
            traces[snr].append(trace)
            traces_az[snr].append(trace_az)

    # Render inferred and simulated maps
    resol = 150
    resol_samples = 80
    maps_sim_rendered = [
        starry_intensity_to_bbtemp(
            m.render(res=resol, projection="Mollweide"), wavelength_grid
        )
        for m in snapshots_maps
    ]
    bbtemp_inferred_rendered = {snr: [] for snr in snr_ratios}
    bbtemp_inferred_samples_rendered = {snr: [] for snr in snr_ratios}

    # Save inferred maps
    map = starry.Map(ydeg_inf)

    for snr in snr_ratios:
        for i in range(len(traces[snr])):
            # Flux
            x_mean = np.mean(traces[snr][i]["w"], axis=0)

            # Mean
            map[1:, :] = x_mean[1:] / x_mean[0]
            map.amp = x_mean[0]
            temp = inferred_intensity_to_bbtemp(
                map.render(res=resol, projection="Mollweide"), filt, params_s, params_p
            )
            bbtemp_inferred_rendered[snr].append(temp)

            # Samples
            samples = []
            for s in np.random.randint(0, draws, 4):
                x_sample = traces[snr][i]["w"][s].reshape(-1)
                map[1:, :] = x_sample[1:] / x_sample[0]
                temp = inferred_intensity_to_bbtemp(
                    map.render(res=resol_samples, projection="Mollweide"),
                    filt,
                    params_s,
                    params_p,
                )
                samples.append(temp)
            bbtemp_inferred_samples_rendered[snr].append(samples)
    fig, ax_sim_maps, ax_text, ax_inf_maps, ax_samples = initialize_figure()

    norm = colors.Normalize(vmin=1000, vmax=1300)

    # Plot simulated map
    map = starry.Map(25)

    for i in range(4):
        map.show(
            image=maps_sim_rendered[i],
            ax=ax_sim_maps[i],
            cmap="OrRd",
            projection="Mollweide",
            norm=norm,
        )

    # Text
    # Build a rectangle in axes coords
    left, width = 0.25, 0.5
    bottom, height = 0.25, 0.5
    right = left + width
    top = bottom + height

    text_list = [
        "Inferred maps - S/N = 20",
        "Inferred maps - S/N = 100",
        "Inferred maps - S/N = 500",
    ]
    for i, a in enumerate(ax_text):
        a.axis("off")
        a.text(
            0.5 * (left + right),
            0.3 * (bottom + top),
            text_list[i],
            horizontalalignment="center",
            verticalalignment="bottom",
            transform=a.transAxes,
            fontweight="bold",
            fontsize=16,
        )

    # Plot inferred maps
    for snr in snr_ratios:
        map = starry.Map(ydeg_inf)
        for i in range(len(traces[snr])):
            map.show(
                image=bbtemp_inferred_rendered[snr][i],
                ax=ax_inf_maps[snr][i],
                cmap="OrRd",
                projection="Mollweide",
                norm=norm,
            )

            # Plot samples
            for s in range(4):
                map.show(
                    image=bbtemp_inferred_samples_rendered[snr][i][s],
                    ax=ax_samples[snr][i][s],
                    cmap="OrRd",
                    projection="Mollweide",
                    norm=norm,
                    grid=False,
                )

    labels = ["$t = 100$ days", "$t = 106$ days", "$t = 108$ days", "$t = 109$ days"]

    for i, a in enumerate(ax_sim_maps):
        a.set_title(labels[i])

    # Colorbar
    cax = fig.add_axes([0.42, 0.08, 0.2, 0.01])
    plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="OrRd"),
        cax=cax,
        orientation="horizontal",
        label="blackbody temperature [K]",
        fraction=1.0,
    )

    for snr in snr_ratios:
        for a in ax_inf_maps[snr]:
            a.set_rasterization_zorder(0)
        ax_samples_flat = [item for sublist in ax_samples[snr] for item in sublist]
        for a in ax_samples_flat:
            a.set_rasterization_zorder(0)

    fig.suptitle("Simulated maps", x=0.517, y=0.97, fontweight="bold", fontsize=16)


# Include everything but the transit
main(t_complete, "hydro_sim_snapshots_fits_complete.pdf")


# Only the eclipse
main(t_eclipse, "hydro_sim_snapshots_fits_eclipse_only.pdf")
