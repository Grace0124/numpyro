import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_csv(
    "https://gist.githubusercontent.com/ucals/"
    "2cf9d101992cb1b78c2cdd6e3bac6a4b/raw/"
    "43034c39052dcf97d4b894d2ec1bc3f90f3623d9/"
    "osic_pulmonary_fibrosis.csv"
)

def chart_patient(patient_id, ax):
    data = train[train["Patient"] == patient_id] # get data for patient_id
    x = data["Weeks"]
    y = data["FVC"] # FVC = forced vital capacity (vol of air exhaled)
    ax.set_title(patient_id)
    sns.regplot(x=x, y=y, ax=ax, ci=None, line_kws = {"color": "red"}) # plot Weeks vs FVC with no confidence interval

f, axes = plt.subplots(1, 3, figsize=(15,5))
chart_patient("ID00007637202177411956430", axes[0])
chart_patient("ID00009637202177434476278", axes[1])
chart_patient("ID00010637202177584971671", axes[2])
# plt.show()

# use partial pooling -> each individual's coefficients come from a common group distribution
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
## cannot do: numpyro.set_platform('gpu')

def model(patient_code, Weeks, FVC_obs=None):
    # priors
    mu_a = numpyro.sample("mu_a", dist.Normal(0, 500))
    sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(100))
    mu_b = numpyro.sample("mu_b", dist.Normal(0, 3))
    sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(3))

    n_patients = len(np.unique(patient_code)) # num of unique patients

    with numpyro.plate("plate_i", n_patients): 
        alpha = numpyro.sample("alpha", dist.Normal(mu_a, sigma_a))
        beta = numpyro.sample("beta", dist.Normal(mu_b, sigma_b))

    sigma = numpyro.sample("sigma", dist.HalfNormal(100))
    FVC_est = alpha[patient_code] + beta[patient_code] * Weeks

    with numpyro.plate("data", len(patient_code)):
        numpyro.sample("obs", dist.Normal(FVC_est, sigma), obs=FVC_obs)

from sklearn.preprocessing import LabelEncoder
patient_encoder = LabelEncoder()
train["patient_code"] = patient_encoder.fit_transform(train["Patient"].values) # number patients starting from 0
FVC_obs = train["FVC"].values
Weeks = train["Weeks"].values
patient_code = train["patient_code"].values


mcmc = MCMC(
    NUTS(model),
    num_warmup = 2000,
    num_samples = 2000
)
mcmc.run(random.PRNGKey(0), patient_code, Weeks, FVC_obs=FVC_obs)
posterior_samples = mcmc.get_samples()