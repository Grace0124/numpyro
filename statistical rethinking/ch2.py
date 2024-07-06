import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

import matplotlib.pyplot as plt
import arviz as az

### I. GRID APPROXIMATION

# grid approximation function
def posterior_grid_binomial(n: int, k: int, s: int) -> jnp.ndarray:
    p_grid: jnp.ndarray = jnp.linspace(0, 1, s) # array of s evenly spaced values from 0 to 1
    priors: jnp.ndarray = jnp.ones(s) # s-dimentional vector of 1s 
    likelihoods: jnp.ndarray =  jnp.exp(dist.Binomial(total_count=9, probs=p_grid).log_prob(6)) # k successes in n trials with p probability of success in each trial
    posteriors: jnp.ndarray = priors * likelihoods 
    posteriors = posteriors / jnp.sum(posteriors) # normalize
    return posteriors

n=9
k=6
s=20
posterior = posterior_grid_binomial(n, k, s)

plt.plot(posterior, "-o")
plt.xlabel("probability of water")
plt.ylabel("posterior probability")
plt.title("20 points")
# plt.show()

### II. QUADRATIC APPROXIMATION

