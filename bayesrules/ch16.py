import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import arviz as az

from jax import random
from jax import vmap
import jax.numpy as jnp 
from jax.scipy.special import logsumexp

import numpyro
from numpyro import handlers
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

### I. LOAD THE DATA
data = pd.read_csv("bayesrules/spotify.csv")
data = data[["artist", "title", "popularity"]]

# calculate mean popularity per artist
mean_popularity = data.groupby("artist")["popularity"].mean().reset_index()
print(mean_popularity.head())
# sort artists by mean popularity
mean_popularity = mean_popularity.sort_values(by="popularity")
# reorder the 'artist' column in data
data["artist"] = pd.Categorical(data["artist"], categories = mean_popularity["artist"], ordered = True)

### II. COMPLETE POOLED MODEL
# total sample size = number of rows = 350
# sns.kdeplot(data, x="popularity")
# plt.show()

popularity = data["popularity"].values

def complete_pooled_model(y=None):
    mu = numpyro.sample("mu", dist.Normal(50, 2.5))
    sigma = numpyro.sample("sigma", dist.Exponential(1))
    numpyro.sample("y", dist.Normal(mu, sigma), obs=y)

mcmc = MCMC(NUTS(complete_pooled_model), 
            num_warmup=2000, 
            num_samples=2000, 
            progress_bar=True)

mcmc.run(random.PRNGKey(4), y=popularity)
mcmc.print_summary()

# check mcmc results with arviz
posterior_samples = mcmc.get_samples()
# x = az.from_numpyro(mcmc)
# az.plot_trace(x, compact=True)
# plt.show()

# plot prediction vs. actual data
predictive = Predictive(complete_pooled_model, posterior_samples)
predictions = predictive(rng_key=random.PRNGKey(2), y=popularity)['y']
print(predictions)

plt.plot(predictions[0])
plt.show()