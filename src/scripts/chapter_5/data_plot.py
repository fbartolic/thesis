import paths
import numpy as np
import os
from astropy.table import Table

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def magnitudes_to_fluxes(mag, mag_err, zeropoint=22):
    flux = 10. ** (0.4 * (zeropoint - mag))
    err_flux = mag_err * flux * np.log(10.) * 0.4
    return flux, err_flux

def get_data(path):
    t = Table.read(os.path.join(path, "phot.dat"), format="ascii")

    # Remove additional columns
    t.columns[0].name = "HJD"
    t.columns[1].name = "mag"
    t.columns[2].name = "mag_err"
    t.keep_columns(("HJD", "mag", "mag_err"))
    return t


lc = get_data(paths.data/"microlensing/blg-0039/")

# Trim
lc['HJD'] = lc['HJD'] - 2450000
lc = lc[lc['HJD'] > 7400]
fobs, ferr = magnitudes_to_fluxes(lc["mag"], lc["mag_err"])
t = lc['HJD']
t, fobs, ferr = np.array(t), np.array(fobs), np.array(ferr)
#
fig, ax = plt.subplots(1, 2, figsize=(12, 5),gridspec_kw={'width_ratios': [2, 1], 'wspace':0.1})

for a in ax:
    a.errorbar(lc['HJD'], lc['mag'], lc['mag_err'],  marker='o', color='k', linestyle='None', alpha=0.2) 
    a.invert_yaxis()
    a.grid(alpha=0.5)
    a.set(xlabel='Time [HJD - 2450000 days]')
    a.xaxis.set_minor_locator(AutoMinorLocator())
    a.yaxis.set_minor_locator(AutoMinorLocator())

    
ax[0].set(ylabel='I magnitude')
ax[0].set_title('OGLE-2016-BLG-0039', pad=20, horizontalalignment='left')
ax[1].set_xlim(7815, 7850)
ax[1].set_yticklabels([])

fig.savefig(paths.figures/"blg-0039.pdf", bbox_inches='tight')
