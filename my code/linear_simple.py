import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS


np.random.seed(42)
theta_true = (25, 0.5)
xdata = 100 * np.random.random(20)
ydata = theta_true[0] + theta_true[1] * xdata

# scatter points
xdata = np.random.normal(xdata, 10)
ydata = np.random.normal(ydata, 10)

plt.plot(xdata, ydata, "ok")
# plt.show()

def linear_model(x, y):
    m = numpyro.sample("slope", dist.Normal(0.5, 0.25))
    b = numpyro.sample("intercept", dist.Normal(25, 5))
    sigma = numpyro.sample("sigma", dist.HalfCauchy(10))

    numpyro.sample("y", dist.Normal(m*x+b, sigma), obs=y)


sampler = MCMC(
    NUTS(linear_model),
    num_warmup=3000,
    num_samples=3000,
    num_chains=1,
    progress_bar=True
)
sampler.run(jax.random.PRNGKey(0), xdata, ydata)

sampler.print_summary()
