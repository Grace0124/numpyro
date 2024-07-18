import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import jax.numpy as jnp
from jax import random, vmap

import numpyro
import numpyro.distributions as dist

### I. SAMPLING FROM GRID-APPROXIMATE POSTERIOR
p_grid = jnp.linspace(0, 1, 1000)
prob_p = jnp.repeat(1, 1000)
# dist.Binomial(9, p_grid) creates binomial distribution with 9 trials and probability of success p_grid
# .log_prob(6) calculates the log of the probability of observing 6/9 successes
# jnp.exp() takes the exponent
prob_data = jnp.exp(dist.Binomial(9, p_grid).log_prob(6)) 
posterior = prob_data * prob_p
posterior = posterior / jnp.sum(posterior)
samples = p_grid[dist.Categorical(probs=posterior).sample(random.PRNGKey(0), (10000,))]

# plot results
# plt.scatter(range(len(samples)), samples, alpha=0.2)
# plt.show()
# az.plot_density({"": samples}, hdi_prob=1)
# plt.show()

### II. SAMPLING TO SUMMARIZE
# posterior prob that the proportion of water < 0.5
jnp.sum(posterior[p_grid < 0.5])
# frequency of parameter values below 0.5
jnp.sum(samples < 0.5) / 1e4
# posterior probability between 0.5 and 0.75
jnp.sum((samples > 0.5) * (samples < 0.75)) / 1e4
# get 80th percentile
jnp.quantile(samples, 0.8)
# get middle 80% between 10th and 90th percentile
jnp.quantile(samples, jnp.array([0.1, 0.9]))

# skewed posterior
p_grid = jnp.linspace(0, 1, 1000)
prior = jnp.ones(1000)
likelihood = jnp.exp(dist.Binomial(3, p_grid).log_prob(3))
posterior = likelihood * prior
posterior = posterior / jnp.sum(posterior)
samples  = p_grid[dist.Categorical(posterior).sample(random.PRNGKey(0), (10000,))]
# 50% CI
jnp.percentile(samples, q=jnp.array([25, 75]))
# 50% HPDI
numpyro.diagnostics.hpdi(samples, 0.5)
# MAP
p_grid[jnp.argmax(posterior)]
# mean and median
jnp.mean(samples)
jnp.median(samples)
# using a loss function
loss = vmap(lambda d: jnp.sum(posterior*jnp.abs(d-p_grid)))(p_grid)
p_grid[jnp.argmin(loss)]

### III. Sampling to simulate prediction
# do 10 simulations
dist.Binomial(total_count=2, probs=0.7).sample(random.PRNGKey(2), (10,))
# do 100,000 simulations with 2 flips each 
dummy_w = dist.Binomial(total_count=2, probs=0.7).sample(random.PRNGKey(2), (100000,))
jnp.unique(dummy_w, return_counts=True)[1] / 1e5
# repeat but with more flips 
dummy_w = dist.Binomial(100, 0.7).sample(random.PRNGKey(2), (100000,))
ax = az.plot_dist(np.asarray(dummy_w), kind="hist", hist_kwargs={"rwidth":0.1})
ax.set_xlabel("dummy water count", fontsize=14)
plt.show()