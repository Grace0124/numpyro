import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoLaplaceApproximation
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO

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

def model(W, L):
    p = numpyro.sample("p", dist.Uniform(0,1)) # uniform prior
    numpyro.sample("W", dist.Binomial(W+L, p), obs=W) # binomial likelihood

guide = AutoLaplaceApproximation(model) # approiximates posterior using a Laplace approximation
svi = SVI(model, guide, optim.Adam(1), Trace_ELBO(), W=6, L=3) # stochastic variational inference
svi_result = svi.run(random.PRNGKey(0), 1000) # run SVI optimization for 1000 iterations
params = svi_result.params

# displaying results
samples = guide.sample_posterior(random.PRNGKey(1), params, sample_shape=(1000,))
numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)
# print results: mean = 0.62, std = 0.14
# interpretation: assuming posterior is Gaussian, it is maximized at 0.67 and has stdev 0.14


