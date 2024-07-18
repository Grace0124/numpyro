import arviz as az
import jax.numpy as jnp
import xarray as xr
az.style.use("arviz-doc")


idata = az.load_arviz_data("centered_eight")
stacked = az.extract(idata)


post = idata.posterior
post["log_tau"] = jnp.log(post["tau"])

# get random subset of samples
az.extract(idata, num_samples=100)

# get a numpy array of values for a given parameter
stacked.mu.values

# compute the posterior mean
post.mean()

