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

# def complete_pooled_model(popularity=None):
#     mu = numpyro.sample("mu", dist.Normal(50, 2.5))
#     sigma = numpyro.sample("sigma", dist.Exponential(1))
#     numpyro.sample("obs", dist.Normal(mu, sigma), obs=popularity)

# mcmc = MCMC(NUTS(complete_pooled_model), 
#             num_warmup=2000, 
#             num_samples=2000, 
#             progress_bar=True)

# mcmc.run(random.PRNGKey(4), popularity=popularity)
# # mcmc.print_summary()

# # check mcmc results with arviz
# posterior_samples = mcmc.get_samples()
# # x = az.from_numpyro(mcmc)
# # az.plot_trace(x, compact=True)
# # plt.show()

# predictive = Predictive(complete_pooled_model, posterior_samples)
# predictions = predictive(random.PRNGKey(0))["obs"]

rng_key = random.PRNGKey(1)

def hierarchical_model(popularity=None):
    mu = numpyro.sample("mu", dist.Normal(50, 2.5))
    sigma_y = numpyro.sample("sigma_y", dist.Exponential(1))
    sigma_u = numpyro.sample("sigma_u", dist.Exponential(1))
    mu_j = numpyro.sample("mu_j", dist.Normal(mu, sigma_u))

    numpyro.sample("obs", dist.Normal(mu_j, sigma_y), obs=popularity)

mcmc = MCMC(
    NUTS(hierarchical_model),
    num_warmup=2000,
    num_samples=2000,
    num_chains=4
)

mcmc.run(rng_key, popularity=popularity)

posterior_samples = mcmc.get_samples()
# idata = az.from_numpyro(mcmc)
# az.plot_trace(idata)
# mcmc.print_summary()
# plt.show()
# plt.clf()

predictive = Predictive(hierarchical_model, posterior_samples = posterior_samples)
predictions = predictive(rng_key, popularity=popularity)

print(predictions)
predicted_popularity = predictions["obs"]
predicted_mean = jnp.mean(predicted_popularity, axis = 0)
predicted_hpdi = hpdi(predicted_popularity, prob=0.9)

plt.figure(figsize=(10, 6))
xticks = jnp.arange(0, 350, 1)
plt.plot(xticks, popularity, label="Actual Popularity", marker="o", linestyle="none")
plt.plot(predicted_popularity, label="Predicted Mean Popularity", color="r")
plt.ylabel("Popularity")
plt.xlabel("Index")
plt.show()