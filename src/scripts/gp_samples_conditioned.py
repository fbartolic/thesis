import paths
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax.config import config

config.update("jax_enable_x64", True)

from tinygp import kernels
from tinygp import GaussianProcess

# Test points
X = np.sort(np.random.default_rng(1).uniform(0, 10, 300))

# Simulate data
random = np.random.default_rng(2)
X_obs = np.sort(random.uniform(2, 8, 8))
y_obs1 = np.sin(2 * X_obs) + 1e-4 * random.normal(size=X_obs.shape)
y_obs2 = np.sin(2 * X_obs) + 1e-1 * random.normal(size=X_obs.shape)


# Condition
kernel = 0.5 * kernels.ExpSquared(scale=1.0)
gp1 = GaussianProcess(kernel, X_obs, diag=1e-4)
gp2 = GaussianProcess(kernel, X_obs, diag=1e-01 ** 2)
cond1 = gp1.condition(y_obs1)
cond2 = gp2.condition(y_obs2)
_, cond_gp1 = gp1.condition(y_obs1, X)
_, cond_gp2 = gp2.condition(y_obs2, X)
y_samp1 = cond_gp1.sample(jax.random.PRNGKey(1), shape=(12,))
y_samp2 = cond_gp2.sample(jax.random.PRNGKey(1), shape=(12,))

fig, ax = plt.subplots(1, 2, figsize=(12, 3), sharey=True, gridspec_kw={"wspace": 0.1})

ax[0].plot(X, y_samp1[0], "k", lw=1.0, alpha=0.5, label="samples")
ax[0].plot(X, y_samp1[1:].T, "k", lw=1.0, alpha=0.5)
ax[0].plot(X_obs, y_obs1, "ko")

ax[1].plot(X, y_samp2[0], "k", lw=1.0, alpha=0.5, label="samples")
ax[1].plot(X, y_samp2[1:].T, "k", lw=1.0, alpha=0.5)
ax[1].plot(X_obs, y_obs2, "ko", label="data")
ax[1].legend(prop={"size": 12}, loc="lower right")

for a in ax:
    a.set(
        xlabel="$x$", ylim=(-2, 2), xlim=(0, 10),
    )

ax[0].set(ylabel="$y=f(x)$")
ax[0].set_title("noiseless observations")
ax[1].set_title("noisy observations")


# Save as pdf
fig.savefig(paths.figures/"gp_samples_conditioned.pdf", bbox_inches="tight")
