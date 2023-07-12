""" This file serves as an example script to perform portfolio sorts."""

import numpy as np
import pandas as pd
from PortSort import PortfolioSort

np.random.seed(321)

# Create random time-series for returns, market value and the characteristic exposure.
# The matrices have N=2,000 assets and T=2,000 weekly observations. In this case, for the ease of an example,
# I take the end-of-week as day for portfolio formation(Sunday).

# Return are drawn from a normal distribution with mu=0.5% and standard deviation sigma=3%.

mu = 0.005
sigma = 0.03
returns = sigma * (np.random.randn(2000 * 2000) / 100) + mu
df_ret = pd.DataFrame(returns.reshape(2000, 2000),
                      index=pd.date_range(end="2022-12-31", periods=2000, freq="w"))

df_mcap = pd.DataFrame(np.random.randn(2000 * 2000).reshape(2000, 2000) * 1e9,
                       index=pd.date_range(end="2022-12-31", periods=2000, freq="w"))

# Calculate a simple momentum strategy over 8-weeks, including the short-term reversal
df_mom = np.log(1 + df_ret).rolling(window=8, min_periods=8).sum()
df_mom = np.exp(df_mom) - 1


# First example: perform a single sort and return each portfolio return and the p-value of a Student's t-test against
# the null hypothesis of mean zero returns.
result = PortfolioSort.single_sort(df_char=df_mom,
                                   df_ret=df_ret,
                                   df_mcap=df_mcap,
                                   quantiles=[0, .2, .4, .6, .8, 1],
                                   speedups=True)

print(result.round(3))

# Instead of returning the p-value of the Student's t-test, return the test statistic itself:
result = PortfolioSort.single_sort(df_char=df_mom,
                                   df_ret=df_ret,
                                   df_mcap=df_mcap,
                                   quantiles=[0, .2, .4, .6, .8, 1],
                                   get_tstat=True)

print(result.round(3))

# Now we are only interested in the sorting results, i.e. which asset was sorted into which portfolio:
result = PortfolioSort.single_sort(df_char=df_mom,
                                   df_ret=df_ret,
                                   df_mcap=df_mcap,
                                   quantiles=[0, .2, .4, .6, .8, 1],
                                   get_quantile_sorts=True)

print(result.head())

# Next, we are not interested in the average portfolio returns, but the return series itself:
result = PortfolioSort.single_sort(df_char=df_mom,
                                   df_ret=df_ret,
                                   df_mcap=df_mcap,
                                   quantiles=[0, .2, .4, .6, .8, 1],
                                   get_series=True)

print(result.head())

# If multiple sorts are performed, consider to use the speedups, which will omit all portfolios except high and low:
result = PortfolioSort.single_sort(df_char=df_mom,
                                   df_ret=df_ret,
                                   df_mcap=df_mcap,
                                   quantiles=[0, .2, .4, .6, .8, 1],
                                   speedups=True)

print(result.round(3))
