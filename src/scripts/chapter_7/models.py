import numpy as np
import starry

import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import *
import arviz as az

import os

numpyro.enable_x64(True)
numpyro.set_host_device_count(4)


starry.config.lazy = False
starry.config.quiet = True


HOMEPATH = os.path.abspath(os.path.split(__file__)[0])


def fit_model_num(
    A,
    fobs,
    ferr,
    ydeg_inf=5,
    tune=500,
    draws=500,
    chains=2,
    sigmoid_constraint=True,
    sigmoid_steepness=1e6,
    sd=1e-02,
):
    ncoeff = (ydeg_inf + 1) ** 2

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

    x_lsq, _ = starry.linalg.solve(A_full, fobs, C=ferr ** 2, L=L,)
    fs_lsq = x_lsq[0]

    # Transform the Ylm coefficients to simplify the structure of the covariance matrix
    # Polar transform
    R = starry.Map(ydeg_inf).ops.dotR(
        np.eye(ncoeff),
        np.array(1.0),
        np.array(0.0),
        np.array(0.0),
        np.array(-np.pi / 2),
    )

    # Transform to group Ylms by order
    idx = [[] for m in range(2 * ydeg_inf + 1)]
    for m in range(ydeg_inf + 1):
        l = np.arange(abs(m), ydeg_inf + 1)
        idx[2 * m] = l ** 2 + l + m
        if m > 0:
            idx[2 * m - 1] = l ** 2 + l - m
    Nb = np.array([len(i) for i in idx], dtype=int)
    ii = np.array([item for sublist in idx for item in sublist], dtype=int)
    G = np.zeros((ncoeff, ncoeff))
    G[np.arange(ncoeff), ii] = 1

    # Full transform
    Q = G @ R
    QInv = np.linalg.inv(Q)

    map = starry.Map(ydeg_inf)
    _, _, Y2P, _, _, _ = map.get_pixel_transforms(oversample=4)

    def model():
        u = []
        j = 0
        for i, nb in enumerate(Nb):
            u.append(
                numpyro.sample(f"u_{i}", dist.Normal(jnp.zeros(nb), sd * jnp.ones(nb)))
            )
            j += nb
        u = jnp.concatenate(u)
        w = numpyro.deterministic("w", jnp.dot(jnp.array(QInv), u))

        if sigmoid_constraint:
            # Penalize values of `p` outside [0, 1]
            p = numpyro.deterministic("p", jnp.dot(jnp.array(Y2P), w)).reshape(-1)
            penalty = -jnp.log(1.0 + jnp.exp(-sigmoid_steepness * p.reshape(-1)))
            # s = 1e3
            # penalty = (
            #    s * p
            #    - jnp.log(jnp.exp(s * p) + 1)
            #    + s * (1 - p)
            #    - jnp.log(jnp.exp(s * (1 - p)) + 1)
            # )
            numpyro.factor("pot", penalty.sum())

        fp = jnp.dot(A_sec, w).reshape(-1)

        fs_delta = numpyro.sample("fs_delta", dist.Normal(0, 1e-05))
        f = (fs_lsq + fs_delta) + fp
        numpyro.deterministic("fpred", f)

        numpyro.sample(
            "obs", dist.Normal(jnp.array(f), jnp.array(ferr)), obs=jnp.array(fobs),
        )

    u_start = 1e-04 * np.random.randn(ncoeff)
    init_vals = {"fs_delta": 0.0, "sd": 1e-04}
    j = 0
    for i, nb in enumerate(Nb):
        init_vals[f"u_{i}"] = u_start[j : j + nb]
        j += nb

    nuts_kernel = NUTS(
        model,
        dense_mass=[(f"u_{i}",) for i in Nb],
        init_strategy=init_to_value(values=init_vals),
        target_accept_prob=0.95,
        max_tree_depth=10,
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=tune,
        num_samples=draws,
        num_chains=chains,
        progress_bar=False,
    )
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key)
    samples = mcmc.get_samples()
    samples_np = {key: np.array(samples[key]) for key in samples.keys()}
    samples_az = az.from_numpyro(
        mcmc, posterior_predictive={"obs": np.array(samples["fpred"])}
    )

    return mcmc, samples_np, samples_az

