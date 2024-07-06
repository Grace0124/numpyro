import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# n = 9
# k = 6
# p = 0.5

# stats.binom.rvs(1, p, size=9)
# result = stats.binom.pmf(k, n, p)
# print(result)


# # grid approximation function
# def posterior_grid_binomial(n: int, k: int, s: int) -> np.ndarray:
#     p_grid = np.linspace(0, 1, s) # array of s evenly spaced values from 0 to 1

#     priors = np.ones(s) # s-dimentional vector of 1s 
#     likelihoods = stats.binom.pmf(k, n, p=p_grid) # k successes in n trials with p probability of success in each trial
#     posteriors = priors * likelihoods 
#     posteriors = posteriors / sum(posteriors) # normalize
#     return posteriors

# n=15
# k=8
# s=50
# posterior = posterior_grid_binomial(n, k, s)

# # create a dataframe from posterior
# aux = pd.DataFrame(posterior).rename({0:'prob'}, axis = 1)
# # add column 'p' with probability values
# aux['p'] = aux.index / 100
# # plot line
# g = sns.lineplot(data=aux, x='p', y='prob')
# # plot scatter plot
# sns.scatterplot(data=aux, x='p', y='prob', ax=g)
# plt.show()


n = 15
k = 8
s = 101
p_grid_1 = np.linspace(0, 1, s)
priors_1 = np.concatenate((np.zeros(50), np.full(51, 0.5)))
likelihoods_1 = stats.binom.pmf(k, n, p = p_grid_1)
posteriors_1 = priors_1 * likelihoods_1
posteriors_1 = posteriors_1 / sum(posteriors_1)
print(posteriors_1)