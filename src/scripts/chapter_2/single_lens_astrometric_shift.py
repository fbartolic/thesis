import paths
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

tau = np.linspace(-1.1, 1.1, 1000)
u0 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

fig, ax = plt.subplots(figsize=(8, 8))

ax.axhline(0, alpha=0.5, color="grey", lw=1)
ax.axvline(0, alpha=0.5, color="grey", lw=1)

for _u0 in u0:
    u = np.linspace(-50.0, 50.0, 5000) + _u0 * 1j

    delta_c_x = (u / (np.abs(u) ** 2 + 2)).real
    delta_c_y = (u / (np.abs(u) ** 2 + 2)).imag

    ax.plot(delta_c_x, delta_c_y, label=f"$u_0=${_u0}", lw=2)

ax.set_aspect(1)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set(xlabel=r"$\delta z_{\mathrm{cent},1}$", ylabel="$\delta z_{\mathrm{cent}, 2}$")
ax.legend(prop={"size": 14})


fig.savefig(paths.figures/"single_lens_astrometric_shift.pdf", bbox_inches="tight")
