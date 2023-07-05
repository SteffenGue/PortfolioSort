""" This file serves as an example script to perform portfolio sorts."""

import numpy as np
import pandas as pd
from PortSort import PortfolioSort

np.random.seed(314)

# Create random time-series for returns, market value and the characteristic exposure.
# The matrices have N=2,000 assets and T=2,000 weekly observations. In this case, for the ease of an example,
# I take the end-of-week (Sunday).

# Return are drawn from a normal distribution with mu=0.1% and standard deviation about 1%.
returns = np.random.randn(2000 * 2000) / 100 + 0.001
df_ret = pd.DataFrame(returns.reshape(2000, 2000),
                      index=pd.date_range(end="2022-12-31", periods=2000, freq="w"))

df_mcap = pd.DataFrame(np.random.randn(2000 * 2000).reshape(2000, 2000) * 1e9,
                       index=pd.date_range(end="2022-12-31", periods=2000, freq="w"))

# Calculate a simple momentum strategy over 8-weeks, including the short-term reversal
df_mom = np.log(1 + df_ret).rolling(window=8, min_periods=8).sum()
df_mom = np.exp(df_mom) - 1

result = PortfolioSort.single_sort(df_char=df_mom,
                                   df_ret=df_ret,
                                   df_mcap=df_mcap,
                                   quantiles=[.2, .4, .6, .8, 1])

print(result.round(3))
