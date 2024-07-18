import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random

### I. GENERATE DATA
from sklearn.datasets import make_sparse_coded_signal

y, X, beta = make_sparse_coded_signal(n_samples=1,
                                   n_components=50,
                                   n_features=100,
                                   n_nonzero_coefs=20,
                                   random_state=0)

X = X.T 
# print(y.shape) # 100 x 1
# print(X.shape) # 100 x 50 
# print(beta.shape) # 50 x 1

print(jnp.equal(y, X@beta))

# add some noise to y
y += 0.1 * random.normal(key=random.PRNGKey(1), shape=(len(y),))

### II. DEFINE MODEL
import numpyro
import numpyro.distributions as dist

def horeshoe_linear_model(y=None, X=None, y_sigma=.1):
    n_predictors = X.shape[1] # 50 predictors (length of beta)
    Tau = numpyro.sample('tau', dist.HalfCauchy(1))
    with numpyro.plate('data', n_predictors): # for i in range(n_predictors):
        Lambda = numpyro.sample('lambda', dist.HalfCauchy(1)) 
        Beta = numpyro.sample('beta', dist.Normal(0, Tau*Lambda))
    mu = jnp.dot(X, Beta) 
    numpyro.sample('y', dist.Normal(mu, y_sigma), obs=y)

### III. Run MCMC
from numpyro.infer import MCMC, NUTS

mcmc = MCMC(
    NUTS(horeshoe_linear_model),
    num_warmup=500,
    num_samples=1000
)
mcmc.run(random.PRNGKey(2), y=y, X=X)

### IV. 
posterior_samples = mcmc.get_samples()
# print(posterior_samples['beta'].shape) # 1000 x 50
# print(posterior_samples['lambda'].shape) # 1000 x 50
# print(posterior_samples['tau'].shape) # 1000 x 1

beta_mu = jnp.mean(posterior_samples['beta'], axis=0) 
print(beta_mu.shape) # 50 x 1
# plt.step(range(len(beta)), beta,  where='mid', lw=1)
# plt.plot(range(len(beta)), beta, '.')
# plt.plot(range(len(beta)), beta_mu, 'g*')
# plt.xlabel(r'$\beta$')
# plt.show()