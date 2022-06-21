import numpy as np
import matplotlib.pyplot as plt


def draw_samples(ndim, nsamples=5000):
    mean = np.zeros(ndim)
    cov = np.diag(np.ones(ndim))
    return np.random.multivariate_normal(mean, cov, nsamples)


distance1 = np.linalg.norm(draw_samples(2), axis=1)
distance2 = np.linalg.norm(draw_samples(10), axis=1)
distance3 = np.linalg.norm(draw_samples(50), axis=1)


fig, ax = plt.subplots(1, 3, figsize=(12, 2), sharey=True)
ax[0].hist(distance1, bins=25, density=True, histtype="step", color="k", lw=2.0)
ax[1].hist(distance2, bins=25, density=True, histtype="step", color="k", lw=2.0)
ax[2].hist(distance3, bins=25, density=True, histtype="step", color="k", lw=2.0)

ax[0].set_title("D = 2")
ax[1].set_title("D = 10")
ax[2].set_title("D = 50")

for a in ax.ravel():
    a.set_xlim(0.0, 10.0)
    a.set_xticks(np.arange(11, step=2))
    a.set_xlabel("Distance from MAP")

# Save as pdf
fig.savefig("typical_sets.pdf", bbox_inches="tight")

