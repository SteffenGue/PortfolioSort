# PortfolioSort
Repository to perform portfolio sorts in empirical asset pricing. This function performs cross-sectional sort of assets,
conditional on a characteristic exposure. The function returns (weighted) portfolio returns, test statistics,
portfolios populations or the portfolios time-series. 

Additionally, overhead is reduced when speedups are activated, which makes this function
applicable for repetitive usage. Bottlenecks are furthermore just-in-time compiled using Numba.


# Example:
 ```
from PortSort import PortfolioSort
 
 
# Generate or import some time series data
returns = <...>
characteristic = <...>
market_value = <...>

# Perform portfolio sorts on a charactistic, using 5 portfolios.
result = PortfolioSort.single_sort(df_char=characteristic,
                                    df_ret=returns,
                                    df_mcap=market_value,
                                    quantiles=[.2, .4, .6, .8, 1])
 
 ```


