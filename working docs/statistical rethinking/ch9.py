import jax.numpy as jnp
from jax import random, lax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_value

import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
import seaborn as sns

### I. Good King Markov
num_weeks = int(1e5)
positions = jnp.repeat(0, num_weeks)
current = 10
def fn(i, val):
    positions, current = val
    positions = positions.at[i].set(current) # basically jax's version of 'positions[i] = current' for arrays
    bern = dist.Bernoulli(0.5).sample(random.fold_in(random.PRNGKey(0), i))
    proposal = current + (bern*2-1)
    proposal = jnp.where(proposal < 1, 10, proposal) # jax version of 'if proposal < 1: proposal = 10'
    proposal = jnp.where(proposal > 10, 1, proposal)
    prob_move = proposal / current
    unif = dist.Uniform().sample(random.fold_in(random.PRNGKey(1), i))
    current = jnp.where(unif < prob_move, proposal, current)
    return positions, current

# for i in range(num_weeks):
#     positions, current = fn(i, (positions, current))
positions, current = lax.fori_loop(0, num_weeks, fn, (positions, current)) # MUCH faster version of for loop above

# plt.plot(range(1, 101), positions[:100], 'o', mfc='none')

plt.hist(positions, bins=range(1, 12), density=True, rwidth=0.1, align="left")
plt.show()

### II. HMC
def get_U(q, a=0, b=1, k=0, d=1) -> int:
    """Calculate the negative log probability at a specific point.
    
    Parameters:
       q:
       a: mean of mu_x
       b: stdev of mu_x
       k: mean of mu_y
       d: stdev of mu_y
    """
    mu_y = q[1]
    mu_x = q[2]
    logprob_y = jnp.sum(dist.Normal(mu_y, 1).log_prob(y))
    logprob_x = jnp.sum(dist.Normal(mu_x, 1).log_prob(x))
    logprob_mu_x = dist.Normal(a, b).log_prob(mu_x)
    logprob_mu_y = dist.Normal(k, d).log_prob(mu_y)
    return -1* (logprob_y + logprob_x + logprob_mu_x + logprob_mu_y)

def get_gradU(q, a=0, b=1, k=0, d=1) -> int:
    """Calculate the value of the gradient of the negative log probability
    function at a specific point.

    Parameters:
       q:
       a: mean of mu_x
       b: stdev of mu_x
       k: mean of mu_y
       d: stdev of mu_y
    """
    mu_y = q[1]
    mu_x = q[2]
    grad_mu_y = jnp.sum(y - mu_y) + (a - mu_y) / b**2
    grad_mu_x = jnp.sum(x - mu_x) + (k - mu_x) / d**2
    return jnp.stack([-grad_mu_y, -grad_mu_x])

with numpyro.handlers.seed(rng_seed=7):
    y = numpyro.sample("y", dist.Normal().expand([50]))
    x = numpyro.sample("x", dist.Normal().expand([50]))
    # normalize
    x = (x-jnp.mean(x)) / jnp.std(x)
    y = (y-jnp.mean(y)) / jnp.std(y)

def HMC(U, grad_U, epsilon, L, current_q, rng):
    """
    
    Parameters:
       # U: function returning negative log-probability of parameter values
       # grad_U: function returning gradient of U at parameter values
       # epsilon: step size
       # L: number of leapfrog steps
       # current_q: current position
    """
    q = current_q
    p = dist.Normal(0, 1).sample(random.fold_in(rng, 0), (q.shape[0],)) # p is momentum
    current_p = p
    # half step for momentum
    p -= epsilon*grad_U(q) / 2
    qtraj = jnp.full((L+1, q.shape[0]), jnp.nan)
    ptraj = qtraj
    qtraj = qtraj.at[0].set(current_q)
    ptraj = ptraj.at[0].set(p)

    # alternate between updating position (q) and momentum (p)
    for i in range(L):
        q += epsilon * p 
        if i != (L-1):
            p -= epsilon*grad_U(q)
            ptraj = ptraj.at[i+1].set(p)
        qtraj = qtraj.at[i+1].set(q)
    
    p -= epsilon*grad_U(q)/2
    ptraj = ptraj.at[L].set(p)
    
    p = -p
    
    # calculate potential + kinetic energy
    current_U = U(current_q)
    current_K = jnp.sum(current_p**2) / 2
    proposed_U = U(q)
    proposed_K = jnp.sum(p**2) / 2

