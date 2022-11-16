import paths
import numpy as np
import starry

import os
from utils import planck

np.random.seed(42)
starry.config.lazy = False
starry.config.quiet = True


def simulation_snapshot_to_ylm(path, wavelength_grid, ydeg=20):
    data = np.loadtxt(path)

    nlat = 512
    nlon = 1024

    temp_grid = data.reshape(nlat, nlon)
    temp_grid = np.roll(temp_grid, int(temp_grid.shape[1] / 2), axis=1)

    x_list = []
    map_tmp = starry.Map(ydeg)

    # Evaluate at fewer points in wavelength for performance reasons
    idcs = np.linspace(0, len(wavelength_grid) - 1, 10).astype(int)
    for i in idcs:
        I_grid = np.pi * planck(temp_grid, wavelength_grid[i])
        map_tmp.load(I_grid, force_psd=False)
        x_list.append(map_tmp._y * map_tmp.amp)

    # Interpolate to full grid
    x_ = np.vstack(x_list).T
    x_interp_list = [
        np.interp(wavelength_grid, wavelength_grid[idcs], x_[i, :])
        for i in range(x_.shape[0])
    ]
    x = np.vstack(x_interp_list)

    return x


input_dir = paths.data/"mapping_exo/hydro_snapshots_raw/T341_1bar/"
output_dir = paths.data/"output/mapping_exo/T341_1bar_ylms/"
fnames = os.listdir(input_dir)

# Sort according to time
vals = np.array([float(fname.split("_")[3]) for fname in fnames])
sorted_idcs = np.argsort(vals)
fnames = np.array(fnames)[sorted_idcs]
vals = vals[sorted_idcs]


# Wavelength grid for starry map (should match filter range)
wavelength_grid = np.linspace(4.5 - 1.2, 4.5 + 1.2, 50)
ydeg = 20



# Iterate over files in order and only select snapshots at the same orbital
# phase 
xs = []
times = []
for fname in fnames:
    time =  float(fname.split("_")[3]) 
    if (time % 1 == 0.) and (len(times) <= 30):
        print(fname)
        times.append(time)
        # Load simulation snapshots as starry maps
        x = simulation_snapshot_to_ylm(
            os.path.join(input_dir, fname), wavelength_grid, ydeg=ydeg
        )
        xs.append(x)

xs = np.stack(xs)
times = np.array(times)
np.save(os.path.join(output_dir, "coefficients.npy"), xs)
np.save(os.path.join(output_dir, "times.npy"), times)

# Save snapshots for single orbit
xs = []
times = []
for fname in fnames:
    time =  float(fname.split("_")[3]) 
    if time >= 25. and time <= 26.:
        print(fname)
        times.append(time)
        # Load simulation snapshots as starry maps
        x = simulation_snapshot_to_ylm(
            os.path.join(input_dir, fname), wavelength_grid, ydeg=ydeg
        )
        xs.append(x)

xs = np.stack(xs)
times = np.array(times)
np.save(os.path.join(output_dir, "coefficients_single_orbit.npy"), xs)
np.save(os.path.join(output_dir, "times_single_orbit.npy"), times)
