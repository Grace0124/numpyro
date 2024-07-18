import jax.numpy as jnp
from jax import random, lax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_value

import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
import seaborn as sns

y = dist.Normal()
print(y.shape())

z = dist.Normal().expand([50])
print(z.shape())
