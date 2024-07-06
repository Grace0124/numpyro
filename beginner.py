# https://num.pyro.ai/en/latest/tutorials/bayesian_regression.html
# jupyter notebook view: https://nbviewer.org/github/pyro-ppl/numpyro/blob/master/notebooks/source/bayesian_regression.ipynb

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from jax import random
from jax import vmap
import jax.numpy as jnp 
from jax.scipy.special import logsumexp

import numpyro
from numpyro import handlers
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

DATASET_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv"
dset = pd.read_csv(DATASET_URL, sep=";")

vars = [
    "Population",
    "MedianAgeMarriage",
    "Marriage",
    "WaffleHouses",
    "South",
    "Divorce",
]

# normalize to N(0,1) for faster results
def standardize(x):
    return (x-x.mean())/x.std()

dset["AgeScaled"] = standardize(dset.MedianAgeMarriage)
dset["MarriageScaled"] = standardize(dset.Marriage)
dset["DivorceScaled"] = standardize(dset.Divorce)



def model(marriage=None, age=None, divorce=None):
    a = numpyro.sample("a", dist.Normal(0.0, 0.2))
    M, A = 0.0, 0.0
    if marriage is not None:
        bM = numpyro.sample("bM", dist.Normal(0.0, 0.5))
        M = bM * marriage
    if age is not None:
        bA = numpyro.sample("bA", dist.Normal(0.0, 0.5))
        A = bA * age
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    mu = a + M + A
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=divorce)


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key) # splits the rng key into two independent rng keys

# Run NUTS.
mcmc = MCMC(
    NUTS(model), 
    num_warmup=1000, 
    num_samples=2000)
mcmc.run(
    rng_key_, marriage=dset.MarriageScaled.values, divorce=dset.DivorceScaled.values
)


mcmc.print_summary() # prints quartiles, eff sample size, r-hat
samples_1 = mcmc.get_samples() 
print(samples_1)

# check results by plotting the regression line
def plot_regression(x, y_mean, y_hpdi):
    """
    x: array of predictor values
    y_mean: array of mean predictions
    y_hpdi: 2D array where the first dimenstion = lower and upper bounds
    
    Returns a graph plotting predictor variable 'x' and repsonse variable 'y_mean' along
    with the HPDI for predictions.
    """
    idx = jnp.argsort(x) # returns indices that would sort array 'x'
    marriage = x[idx] # sorted predictor values
    mean = y_mean[idx] # sorted mean predictions
    hpdi = y_hpdi[:, idx] # sorted by 'x'
    divorce = dset.DivorceScaled.values[idx] # sorted values from the dataset

    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(6,6))
    ax.plot(marriage, mean) # plot predictions
    ax.plot(marriage, divorce, "o") # plot actual values
    ax.fill_between(marriage, hpdi[0], hpdi[1], alpha=0.3, interpolate=True) # fill in area btwn lower and upper bounds of the confidence interval
    return ax


posterior_mu = (
    # expand_dims adds a new axis to the arrays... not sure why lol
    # perhaps so that the multiplication with dset.MarriageScaled has the right dimensions??
    jnp.expand_dims(samples_1["a"], -1) 
    + jnp.expand_dims(samples_1["bM"], -1) * dset.MarriageScaled.values
)

mean_mu = jnp.mean(posterior_mu, axis=0)
hpdi_mu = hpdi(posterior_mu, 0.9) #90% HPDI
ax = plot_regression(dset.MarriageScaled.values, mean_mu, hpdi_mu)
ax.set(
    xlabel="Marriage rate", ylabel="Divorce rate", title="Regression line with 90% CI"
);
plt.show()

# Sample from prior predictive distribution
from numpyro.infer import Predictive

rng_key, rng_key_ = random.split(rng_key)
prior_predictive = Predictive(model, num_samples=100) # a predictive distribution sampler
prior_predictions = prior_predictive(rng_key_, marriage=dset.MarriageScaled.values)["obs"] # draws samples from prior predictive using the rng key 
mean_prior_pred = jnp.mean(prior_predictions, axis=0)
hpdi_prior_pred = hpdi(prior_predictions, 0.9)

ax = plot_regression(dset.MarriageScaled.values, mean_prior_pred, hpdi_prior_pred)
plt.show()