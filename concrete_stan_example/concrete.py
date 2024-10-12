import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import gamma, digamma
import xarray as xr
from cmdstanpy import CmdStanModel, from_csv, write_stan_json

args = {}
args['seed'] = 1234
args['main'] = Path(__file__).resolve()
args['cwd'] = args['main'].parent.resolve()
args['stan_dir'] = args['cwd'] / 'stan'
args['stanc_args'] = {"include-paths": [str(args['cwd'] / 'src')]}
args['hpp'] = args['cwd'] / 'src' / 'concrete.hpp'


def concrete_rng(alpha, lambd, size=1, seed=None):
    if seed is not None:
        np.random.seed(seed) 
    n_size = len(alpha)
    weights = np.zeros((size, n_size))
    for _ in range(size):
        uniform_random = np.random.uniform(0.0, 1.0, n_size)
        gumble = -np.log(-np.log(uniform_random))
        logit = (np.log(alpha) + gumble) / lambd
        weights_sample = np.exp(logit)
        weights_sample /= np.sum(weights_sample)
        weights[_] = weights_sample
    return weights if size > 1 else weights[0]



data = concrete_rng(alpha=np.array([2.0, 3.0]), lambd=1, size=1000, seed=args['seed'])
stan_data_0 = {
    "N": len(data),
    "n": 2,
    "y": data,
}

stan_file = args['stan_dir'] / 'concrete.stan'
model = CmdStanModel(stan_file=stan_file, force_compile=True, 
                    user_header=args['hpp'], stanc_options=args['stanc_args'])
fit = model.sample(data=stan_data_0,
                    **{"chains": 4, "iter_warmup": 500, "iter_sampling": 500, 
                    "show_console": True, "seed": args['seed'], "refresh": 20,}
                    )
print(fit.diagnose())

fit.stan_variable("alpha").mean(axis=0)
fit.stan_variable("lambda").mean(axis=0)






# -------------------------- Verify logp and derivative --------------------------
def concrete_pdf(x, alpha, lambd):
    n = len(x)
    factorial_term = gamma(n)  # (n-1)!
    lambda_term = lambd**(n - 1)  # λ^(n-1)
    denominator = np.sum(alpha * x**(-lambd))
    product_term = np.prod(alpha * x**(-lambd - 1) / denominator)
    result = factorial_term * lambda_term * product_term
    return result


x = np.array([0.5, 0.5])
alpha = np.array([2,3])
lambd = 1


concrete_pdf(x, alpha, lambd)


x = np.array([0.389018, 0.610982])
alpha = np.array([0.144838, 1.22924])
lambd = 1.14761


x = np.array([0.548653, 0.451347])
alpha = np.array([0.208797, 5.37799])
lambd = 0.723815


def derivatives(x, alpha, lambd):
    n = len(x)
    S = np.sum(alpha * x**(-lambd))
    
    dS_dlambda = -np.sum(alpha * x**(-lambd) * np.log(x))
    d_log_p_dlambda = (n - 1) / lambd - np.sum(np.log(x)) - n * dS_dlambda / S
    
    dS_dalpha = x**(-lambd)
    d_log_p_dalpha = 1 / alpha - n * dS_dalpha / S
    
    dS_dx = -alpha * lambd * x**(-lambd-1)
    d_log_p_dx = -(lambd + 1) / x - n * dS_dx / S
    
    return d_log_p_dx, d_log_p_dalpha, d_log_p_dlambda

derivatives(x, alpha, lambd)




