import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

size = 200
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
true_regression_line = true_intercept + true_slope * x 
# noise
y = true_regression_line + np.random.normal(0, 0.5, size)

plt.scatter(x, y, alpha=0.8)
plt.plot(x, true_regression_line, c='r', label="True Regression Line")
plt.legend();

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import jax.numpy as jnp
from jax import random
import arviz as az

def numpyro_model(x, y):
    # priors; current choice of numbers is arbitrary
    intercept = numpyro.sample("alpha", dist.Normal(0, 20))
    slope = numpyro.sample("beta", dist.Normal(0, 20))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(10))

    # deterministic var for mu
    mu = numpyro.deterministic('mu', intercept + slope * x)

    # y_hat ~ N(mu, sigma), y_hat is an observed variable
    likelihood = numpyro.sample("y", dist.Normal(mu, sigma), obs = y)


# model
numpyro.render_model(numpyro_model, model_args=(x,y))

# create an 'MCMC' object
mcmc = MCMC(sampler=NUTS(numpyro_model), 
            num_warmup = 1000,
            num_samples = 1000,
            num_chains = 4)

# run MCMC sampler
mcmc.run(rng_key=random.PRNGKey(seed=42),
         x=x,
         y=y)

mcmc.print_summary()

posterior_samples = mcmc.get_samples()

predictive = Predictive(numpyro_model, posterior_samples = posterior_samples)
posterior_predictive = predictive(random.PRNGKey(1), x=x, y=None)

# convert to arviz.InferenceData obj
idata = az.from_numpyro(mcmc, posterior_predictive=posterior_predictive)
az.plot_trace(mcmc, figsize=(9,9))
plt.show()