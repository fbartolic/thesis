import paths
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax.config import config

config.update("jax_enable_x64", True)

from tinygp import kernels
from tinygp import GaussianProcess


# Evaluation points
X = np.sort(np.random.default_rng(1).uniform(0, 10, 300))
gp1 = GaussianProcess(kernels.ExpSquared(scale=0.5), X)
gp2 = GaussianProcess(kernels.ExpSquared(scale=2.0), X)

y1 = gp1.sample(jax.random.PRNGKey(4), shape=(10,))
y2 = gp2.sample(jax.random.PRNGKey(4), shape=(10,))

fig, ax = plt.subplots(1, 2, figsize=(12, 3), sharey=True, gridspec_kw={"wspace": 0.1})
ax[0].plot(X, y1.T, color="k", lw=1.0, alpha=0.7)
ax[1].plot(X, y2.T, color="k", lw=1.0, alpha=0.7)

for a in ax:
    a.set(
        xlabel="$x$", ylim=(-3, 3), xlim=(0, 10),
    )

ax[0].set(ylabel="$y=f(x)$")
ax[0].set_title("$l=0.5$")
ax[1].set_title("$l=2.0$")

# Save as pdf
fig.savefig(paths.figures/"gp_samples.pdf", bbox_inches="tight")
