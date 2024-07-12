import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import BSpline
from scipy.stats import gaussian_kde

import jax.numpy as jnp
from jax import random, vmap

import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.diagnostics import hpdi, print_summary
from numpyro.infer import Predictive, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoLaplaceApproximation

# ### I. Normal Distribution by Addition
# ## take 1000 samples of 16 elements each
# ## sum up the elements in each sample to get positions
# position = jnp.sum(dist.Uniform(-1, 1).sample(rng_key, (1000, 16)), -1)

# ### II. Normal Distribution by Multiplication
# small_growth = jnp.prod(dist.Uniform(1.0, 1.1).sample(rng_key, (10000, 12)), -1)
# big_growth = jnp.prod(dist.Uniform(1.0, 1.5).sample(rng_key, (10000, 12)), -1)

# ## plot sample results
# az.plot_density({"small growth": small_growth}, hdi_prob=1)
# x = jnp.sort(small_growth)
# ## plot PDF of normal distribution
# plt.plot(x, jnp.exp(dist.Normal(jnp.mean(x), jnp.std(x)).log_prob(x)), '--')
# plt.show()
# az.plot_density({"big growth": big_growth}, hdi_prob=1)
# y = jnp.sort(big_growth)
# plt.plot(y, jnp.exp(dist.Normal(jnp.mean(y), jnp.std(y)).log_prob(y)), '--')
# plt.show()

# ### III. Normal Distribution by Log-Multiplication
# log_big = jnp.log(jnp.prod(1 + dist.Uniform(0, 0.5).sample(rng_key, (10000, 12)), -1))

### IV. Gaussian Model of Height

howell1 = pd.read_csv("data/howell1.csv", sep=",")
d = howell1 # columns: height, weight, age, male; 544 rows

# print_summary(dict(zip(d.columns, d.T.values)), 0.89, False)

d2 = d[d["age"] >= 18]

# az.plot_density({"height": d2["height"]}, hdi_prob=1)

# x1 = jnp.linspace(100, 250, 101)
# plt.plot(x1, jnp.exp(dist.Normal(178, 20).log_prob(x)))
# x2 = jnp.linspace(-10, 60, 101)
# plt.plot(x2, jnp.exp(dist.Uniform(0, 50, validate_args=True).log_prob(x2)))
# plt.show()

## Simulate the Prior Predictive
# sample_mu = dist.Normal(178, 20).sample(random.PRNGKey(1), (10000,))
# sample_sigma = dist.Uniform(0,50).sample(random.PRNGKey(2), (10000,))
# prior_h = dist.Normal(sample_mu, sample_sigma).sample(random.PRNGKey(3))
# az.plot_kde(prior_h)
# plt.show()

# ## Find the posterior using quadratic approximation
# def flist(height):
#     mu = numpyro.sample("mu", dist.Normal(178, 20))
#     sigma =numpyro.sample("sigma", dist.Uniform(0, 50))
#     numpyro.sample("height", dist.Normal(mu, sigma), obs=height)

# m4_1 = AutoLaplaceApproximation(flist)
# svi = SVI(flist, m4_1, optim.Adam(1), Trace_ELBO(), height=d2.height.values)
# svi_result = svi.run(random.PRNGKey(0), 2000)
# p4_1 = svi_result.params
# samples = m4_1.sample_posterior(random.PRNGKey(1), p4_1, sample_shape=(1000,))
# print_summary(samples, 0.89, False)
# post = m4_1.sample_posterior(random.PRNGKey(1), p4_1, sample_shape=(10000,))

# plt.plot(d2.height, d2.weight, 'o', linestyle='None')
# plt.xlabel("Height (cm)")
# plt.ylabel("Weight (kg)")
# plt.show()

# with numpyro.handlers.seed(rng_seed=2971):
#     N = 100
#     a = numpyro.sample("a", dist.Normal(178, 20).expand([N]))
#     b = numpyro.sample("b", dist.LogNormal(0, 1).expand([N]))

# plt.subplot()
# plt.xlim((min(d2.weight), max(d2.weight)))
# plt.ylim(-100, 400)
# plt.xlabel("weight")
# plt.ylabel("height")
# xbar = d2.weight.mean()
# x = jnp.linspace(min(d2.weight), max(d2.weight), 101)
# for i in range(N):
#     plt.plot(x, a[i] + b[i] * (x-xbar), "k", alpha=0.2)
# plt.show()

xbar = d2.weight.mean()

def model(weight, height):
    sigma = numpyro.sample("sigma", dist.Uniform(0, 50))
    beta = numpyro.sample("beta", dist.LogNormal(0, 1))
    alpha = numpyro.sample("alpha", dist.Normal(178, 20))
    mu_i = numpyro.deterministic("mu_i", alpha+beta*(weight - xbar))
    numpyro.sample("height", dist.Normal(mu_i, sigma), obs=height)

