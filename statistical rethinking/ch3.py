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

# I. SAMPLING FROM GRID-APPROXIMATE POSTERIOR
p_grid = jnp.linspace(0, 1, 1000)
prob_p = jnp.repeat(1, 1000)
# dist.Binomial(9, p_grid) creates binomial distribution with 9 trials and probability of success p_grid
# .log_prob(6) calculates the log of the probability of observing 6/9 successes
# jnp.exp() takes the exponent
prob_data = jnp.exp(dist.Binomial(9, p_grid).log_prob(6)) 
posterior = prob_data * prob_p
posterior = posterior / jnp.sum(posterior)
samples = p_grid[dist.Categorical(probs=posterior).sample(random.PRNGKey(0), (10000,))]