import numpy as np
import matplotlib.pyplot as plt

# # EXAMPLE 1: LINEAR REGRESSION WITH OUTLIERS

# outlier probability = 0.8
true_frac = 0.8

# slope = 1, intercept = 0
true_params = [1.0, 0.0]

# outliers ~ N(0, 1)
true_outliers = [0.0, 1.0]

# generate data
np.random.seed(12)
x = np.sort(np.random.uniform(-2, 2, 15))
yerr = 0.2 * np.ones_like(x)
y = true_params[0]* x + true_params[1] + yerr * np.random.randn(len(x))

# replace some points with outliers
m_bkg = np.random.rand(len(x)) > true_frac
y[m_bkg] = true_outliers[0]
y[m_bkg] += np.sqrt(true_outliers[1] + yerr[m_bkg]**2) * np.random.randn(sum(m_bkg))

# save true line
x0 = np.linspace(-2.1, 2.1, 200)
y0 = np.dot(np.vander(x0, 2), true_params)

def plot_data():
    plt.errorbar(x, y, yerr=yerr, fmt=",k")
    plt.scatter(x[m_bkg], y[m_bkg])
    plt.scatter(x[~m_bkg], y[~m_bkg])
    plt.plot(x0, y0)
    plt.xlim(-2.1, 2.1)
    plt.ylim(-2.5, 2.5)
    plt.show()

# plot_data()

import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

# set number of GPUs to use
numpyro.set_host_device_count(2)

def linear_model(x, yerr, y=None):
    # define priors
    theta = numpyro.sample("theta", dist.Uniform(-0.5*jnp.pi, 0.5*jnp.pi))
    b_perp = numpyro.sample("b_perp", dist.Normal(0, 1))

    # transformed parameters
    m = numpyro.deterministic("m", jnp.tan(theta))
    b = numpyro.deterministic("b", b_perp / jnp.cos(theta))

    # numpyro.plate is similar to a loop, but performs parallel operations efficiently
    with numpyro.plate("data", len(x)):
        numpyro.sample("y", dist.Normal(m*x+b, yerr), obs=y)

# use NUTS to sample from posterior
sampler = MCMC(
    NUTS(linear_model),
    num_warmup=2000, # warmup steps (not used in final analysis)
    num_samples=2000, # num samples after warmup
    num_chains=2, # num MCMC chains to run in parallel
    progress_bar=True
)
sampler.run(jax.random.PRNGKey(0), x, yerr, y=y)

# now let's check the convergence with ArviZ
import arviz as az
inf_data = az.from_numpyro(sampler)
summary = az.summary(inf_data)
print(summary) # r_hat = 1.0 --> each sample is independent yay