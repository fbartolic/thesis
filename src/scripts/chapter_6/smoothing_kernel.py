import paths
import numpy as np
import starry
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from  io_mapping_utils import get_smoothing_filter

np.random.seed(42)
starry.config.lazy = False

# Dummy featureless map
map0 = starry.Map(ydeg=1)
map0.show()

# Compute the intensity along the equator
# Remove the baseline intensity and normalize it
lon = np.linspace(-90, 90, 300)
baseline = 1.0 / np.pi

# Compute the intensity of a normalized gaussian
# in cos(longitude) with the same standard deviation
coslon = np.cos(lon * np.pi / 180)
spot_ang_dim = 5 * np.pi / 180
spot_sigma = 1 - np.cos(spot_ang_dim / 2)
I_gaussian = np.exp(-((coslon - 1) ** 2) / (2 * spot_sigma ** 2))

degs = [10, 15, 20]

Is = np.zeros((3, len(degs), 300))
smoothing_sigmas = np.array([0.0, 0.1, 0.2])


def get_map(deg, sig_smooth, sig_size):
    if sig_smooth == 0:
        map = starry.Map(deg)
        map.add_spot(amp=2.0, sigma=sig_size, lat=0.0, lon=0, relative=False)
        map.amp = 20
    else:
        map = starry.Map(deg)
        map.add_spot(amp=2.0, sigma=sig_size, lat=0.0, lon=0, relative=False)
        map.amp = 20
        S = get_smoothing_filter(deg, sig_smooth)
        x = map.amp * map.y
        x_smooth = (S @ x[:, None]).reshape(-1)
        map[:, :] = x_smooth / x_smooth[0]
        map.amp = x_smooth[0]

    return map


for j in range(len(smoothing_sigmas)):
    for l in range(len(degs)):
        map = get_map(degs[l], smoothing_sigmas[j], spot_sigma)
        I = (map.intensity(lon=lon) - baseline) / (
            map.intensity(lon=0) - baseline
        )
        Is[j, l] = I


fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
fig.subplots_adjust(wspace=0.15, hspace=0.1)

for i in range(len(smoothing_sigmas)):
    for l in range(len(degs)):
        ax[i].plot(
            lon, Is[i, l], alpha=0.8, color=f"C{l}", label=f"$l = {degs[l]}$"
        )
    ax[i].plot(lon, I_gaussian, "k-", alpha=0.8, label="exact")
    ax[i].set_title(r"$\sigma_s={}$".format(smoothing_sigmas[i]))

for a in ax.flatten():
    a.set_xticks(np.arange(-60, 90, 30))
    a.set_xlim(-60, 60)
    a.grid(alpha=0.5)
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())
#     a.grid()

ax[-1].legend(loc=1, prop={"size": 11.2})
fig.text(0.5, -0.08, "Longitude [deg]", ha="center")
fig.text(0.04, 0.5, "Normalized intensity", va="center", rotation="vertical")

# Save
fig.savefig(paths.figures/"smoothing_kernel.pdf", bbox_inches="tight", dpi=200)
