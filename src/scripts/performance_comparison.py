import paths
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
from jax import random

import matplotlib.pyplot as plt

from caustics import (
    critical_and_caustic_curves,
    mag_extended_source,
)

from caustics.utils import *
from caustics.integrate import *

import MulensModel as mm

import timeit

def mag_vbb_binary(w0, rho, a, e1, u1=0., accuracy=1e-05):
    e2 = 1 - e1
    x_cm = (e1 - e2)*a
    bl = mm.BinaryLens(e2, e1, 2*a)
    return bl.vbbl_magnification(w0.real - x_cm, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1)

def mag_ac_binary(w0, rho, a, e1, u1=0., accuracy=0.05, ld_accuracy=1e-03):
    e2 = 1 - e1
    x_cm = (e1 - e2)*a
    bl = mm.BinaryLens(e2, e1, 2*a)
    if u1==0.:
        return bl.adaptive_contouring_magnification(
            w0.real - x_cm, w0.imag, rho, accuracy=accuracy, 
        )
    else:
        return bl.adaptive_contouring_magnification(
            w0.real - x_cm, w0.imag, rho, accuracy=accuracy, u_limb_darkening=u1, ld_accuracy=ld_accuracy
        )

def time(func, number=5, repeats=2):
    times = 1000*np.array((timeit.Timer(func).repeat(repeat=repeats, number=number)))/number
    return np.mean(times)

# Parameters
a, e1 = 0.45, 0.8
u1 = 0.7

npts = 3 # dozen points
critical_curves, caustic_curves = critical_and_caustic_curves(
    npts=npts, nlenses=2, a=a, e1=e1
)
caustic_curves = caustic_curves.reshape(-1)

# Adjusted such that all codes are roughly equivalent in terms of accuracy
acc_vbb = 6e-03
acc_ac = 1e-02
acc_ac_ld = 1e-03

npts_limb = 500
npts_ld = 100

rho_list = [1., 1e-01, 1e-02, 1e-03]

t_list = []
t_vbb_list = []
t_ac_list = []

t_list_ld = []
t_vbb_list_ld = []
t_ac_list_ld = []

for rho in rho_list:
    print("rho = ", rho)

    # Compile
    mag_binary = lambda w: mag_extended_source(
        w,
        rho,
        nlenses=2,
        npts_limb=npts_limb,
        a=a,
        e1=e1,
    )
    mag_binary_ld = lambda w: mag_extended_source(
        w,
        rho,
        nlenses=2,
        npts_limb=npts_limb,
        limb_darkening=True,
        u1=u1,
        npts_ld=npts_ld,
        a=a,
        e1=e1,
    )

    # Generate random test points near the caustics
    key = random.PRNGKey(42)
    key, subkey1, subkey2 = random.split(key, num=3)
    phi = random.uniform(subkey1, caustic_curves.shape, minval=-np.pi, maxval=np.pi)
    r = random.uniform(subkey2, caustic_curves.shape, minval=0., maxval=2*rho)
    w_test = caustic_curves.reshape(-1) + r*np.exp(1j*phi)

    # Caustics
    print(mag_binary(w_test[0]).block_until_ready())
    print(mag_binary_ld(w_test[0]).block_until_ready())

    t_caustics = np.sum([
        time(lambda: mag_binary(w).block_until_ready()) for w in w_test
    ])
    t_caustics_ld = np.sum([
        time(lambda: mag_binary_ld(w).block_until_ready()) for w in w_test
    ])

    # VBB
    t_vbb = np.sum([time(lambda: mag_vbb_binary(w, rho, a, e1, u1=0.0, accuracy=1e-04)) for w in w_test])
    t_vbb_ld = np.sum([time(lambda: mag_vbb_binary(w, rho, a, e1, u1=u1, accuracy=1e-04)) for w in w_test])

    # AC
    t_ac = np.sum([time(lambda: mag_ac_binary(w, rho, a, e1, u1=0.0, accuracy=acc_ac)) for w in w_test])
    t_ac_ld = np.sum([time(lambda: mag_ac_binary(w, rho, a, e1, u1=u1, accuracy=acc_ac, ld_accuracy=acc_ac_ld)) for w in w_test])

    t_list.append(t_caustics)
    t_vbb_list.append(t_vbb)
    t_ac_list.append(t_ac)

    t_list_ld.append(t_caustics_ld)
    t_ac_list_ld.append(t_ac_ld)
    t_vbb_list_ld.append(t_vbb_ld)

# https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
def subcategorybar(ax, X, vals,  labels, colors, width=0.7):
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(len(vals)):
        ax.bar(
            _X - width/2. + i/float(n)*width, 
            vals[i], 
            width=width/float(n), 
            align="edge",
            alpha=0.75,
            label=labels[i],
            color=colors[i]
        )   
    ax.set_xticks(_X, X)

labels = [
    r"$\rho_\star=10^{0}$", r"$\rho_\star=10^{-1}$", r"$\rho_\star=10^{-2}$", r"$\rho_\star=10^{-3}$", 
]
X = ['caustics', 'VBBinaryLensing', 'Adaptive\nContouring']

vals1 = [
    [t_list[0], t_vbb_list[0], t_ac_list[0]],
    [t_list[1], t_vbb_list[1], t_ac_list[1]],
    [t_list[2], t_vbb_list[2], t_ac_list[2]],
    [t_list[3], t_vbb_list[3], t_ac_list[3]],
]
vals2 = [
    [t_list_ld[0], t_vbb_list_ld[0], t_ac_list_ld[0]],
    [t_list_ld[1], t_vbb_list_ld[1], t_ac_list_ld[1]],
    [t_list_ld[2], t_vbb_list_ld[2], t_ac_list_ld[2]],
    [t_list_ld[3], t_vbb_list_ld[3], t_ac_list_ld[3]],
]

import matplotlib
cmap = matplotlib.cm.get_cmap('RdPu')
colors = cmap(np.linspace(0.2, 1, 4))

fig, ax = plt.subplots(1, 2, figsize=(14,6), sharey=True,gridspec_kw={'wspace':0.1})
subcategorybar(ax[0], X, np.array(vals1)/12, labels, colors)
subcategorybar(ax[1], X, np.array(vals2)/12, labels, colors)

for _a in ax:
    _a.set_yscale('log')
    _a.grid(alpha=0.5, zorder=-1)
    _a.set_axisbelow(True)

ax[0].legend(fontsize=14, loc="upper left")
ax[0].set_ylabel('Evaluation time [ms]')
ax[0].set_title("Uniform brightness source")
ax[1].set_title("Limb-darkened source")
ax[1].set_yticks([1e0, 1e1, 1e2, 1e3, 1e4])

fig.savefig(paths.figures/"performance_comparison.pdf", bbox_inches="tight")
