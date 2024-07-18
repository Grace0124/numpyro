import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import jax.numpy as jnp
from jax import random, lax
from jax.scipy.special import expit

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import effective_sample_size
from numpyro.infer import MCMC, NUTS, Predictive

reedfrogs = pd.read_csv("data/reedfrogs.csv", sep=";")
d = reedfrogs

# number the rows 0 to 47
d["tank"] = jnp.arange(d.shape[0])
# store table columns in a dictionary for easier access
# S: number surviving
# N: density
# tank: tank number (0 to 47)
dat = dict(S = d.surv.values,
           N = d.density.values,
           tank = d.tank.values)

# posterior: number surviving (S)

# model 1: no pooling
def model_1(tank, N, S):
    a = numpyro.sample("a", dist.Normal(0, 1.5), sample_shape= tank.shape)
    logit_p = a[tank]
    numpyro.sample("S", dist.Binomial(N, logits=logit_p), obs=S)

mcmc_1 = MCMC(
    NUTS(model_1),
    num_warmup=500,
    num_samples=500,
)
mcmc_1.run(random.PRNGKey(1), **dat) # figure out this operator ** 
# mcmc_1.print_summary()

# x = az.from_numpyro(mcmc_1)
# az.plot_trace(x)
# plt.show()

# model 2: partial pooling
def model_2(tank, N, S):
    a_bar = numpyro.sample("a_bar", dist.Normal(0, 1.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    a = numpyro.sample("a", dist.Normal(a_bar, sigma), sample_shape = tank.shape)
    logit_p = a[tank]
    numpyro.sample("S", dist.Binomial(N, logits=logit_p), obs=S)

mcmc_2 = MCMC(
    NUTS(model_2),
    num_warmup=500,
    num_samples=500,
)
mcmc_2.run(random.PRNGKey(1), **dat)
# mcmc_2.print_summary()

# # mcmc_2 is rank 0, so it is better than mcmc_1
# x = az.compare({
#     "mcmc_1": az.from_numpyro(mcmc_1),
#     "mcmc_2": az.from_numpyro(mcmc_2)
#     },
#     ic="waic",
#     scale="deviance"
# )

posterior = mcmc_2.get_samples()
idata = az.from_numpyro(mcmc_2)
az.plot_trace(mcmc_2)
plt.show()

plt.clf()
# raw proportions
d["propsurv.est"] = expit(jnp.mean(posterior["a"], 0)) # logistic sigmoid function of each tank's median
print(d.head())
plt.plot(jnp.arange(1, 49), d.propsurv, "o", alpha=0.5)
plt.xlabel("tank number")
plt.ylabel("proportion survived")

# posterior means
plt.plot(jnp.arange(1, 49), d["propsurv.est"], "ko", mfc="w")
plt.show()

