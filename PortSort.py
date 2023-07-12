from typing import Union, Optional

from numba import jit
from numba.typed import List

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp as tt


class PortfolioSort:
    """
    Class to perform single and double portfolio sorts
    """

    @staticmethod
    def single_sort(df_char: pd.DataFrame,
                    df_ret: pd.DataFrame,
                    df_mcap: pd.DataFrame,
                    quantiles: Union[list, tuple, int],
                    min_assets: int = 10,
                    value_weighted: Union[bool, float] = True,
                    get_series: bool = False,
                    get_quantile_sorts: bool = False,
                    get_tstat: bool = False,
                    char_lag: int = -1,
                    speedups: bool = False,
                    **kwargs) -> Union[tuple, pd.DataFrame]:
        """ Performs single portfolio sorts based on a characteristic and creates the long-short portfolio.

            :param df_char: TxN matrix of characteristic exposures (not lagged).
            :param df_ret: TxN matrix of returns (not lagged).
            :param df_mcap: TxN matrix of market capitalization (not lagged).
            :param quantiles: List/tuple of portfolio percentiles.
            :param min_assets: Minimum required assets per portfolio.
            :param value_weighted: Portfolio weighting schema.
            :param get_series: Return the portfolio timeseries instead of the average.
            :param get_quantile_sorts: Return the portfolio sorts.
            :param get_tstat: Return the t-statistic instead of the p-values.
            :param breakpoints: Custom define breakpoint calculations as fraction of cumulative market cap.
            :param char_lag: The lag between characteristic and return calculation.
            :param speedups: Enable speedups. This will ignore all quantiles except from the low and high portfolio.

            @return The portfolio means and test statistics as a tuple of Numpy arrays, or a Pandas DataFrame.
        """

        # Check the quantiles input, and delete 0 if included
        quantiles = [x for x in quantiles if x != 0]

        # Set the minimum required asset per timestamp
        minObs = len(quantiles) * min_assets

        # Delete np.inf
        df_char.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_ret.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Retrieve mutually shared columns and indexes
        timestamps, columns = PortfolioSort._get_mutual_index(minObs, *[df_char, df_ret, df_mcap])
        df_ret = df_ret.loc[timestamps, columns].to_numpy()
        df_char = df_char.loc[timestamps, columns]
        df_mcap = df_mcap.loc[timestamps, columns]

        # Delete all observations with any kind of missing value before percentile calculation and portfolio formation.
        # A valid observation consists of non-missing values for: char (t+0), market value (t+0), and returns (t+1).
        # Attention: This effectively introduces a forward-looking bias, as the returns are leading!
        valid_observations = PortfolioSort._get_valid_observations(df_char,
                                                                   np.roll(df_ret, char_lag, 0),
                                                                   df_mcap)

        df_char[~valid_observations] = np.nan
        df_ret[~valid_observations] = np.nan
        df_mcap[~valid_observations] = np.nan

        # Determine if NaNs are present in the return matrix in order to count the portfolio population correctly.
        # Otherwise, np.unique() would count NaNs as an additional portfolio.
        if np.isnan(df_ret).any():
            PortfolioSort.has_nan = True
        else:
            PortfolioSort.has_nan = False

        # Check length of columns and indexes
        if not df_char.shape == df_ret.shape == df_mcap.shape:
            raise ValueError("Input shapes do not match: %s-%s-%s" % (df_char.shape, df_ret.shape, df_mcap.shape))

        # Calculate the breakpoints for the portfolios
        breakpoints = PortfolioSort._get_breakpoints(df_char.to_numpy(), List(quantiles))

        # Assign quantile values to all observations.
        try:
            quantile_sorts = PortfolioSort._assign_quantiles(df_char.to_numpy(), breakpoints)
            quantile_sorts = np.stack(quantile_sorts, axis=0)
        except ValueError:
            return pd.DataFrame()

        if get_quantile_sorts:
            # Return the portfolio occupation if specified
            return pd.DataFrame(quantile_sorts, index=timestamps, columns=columns).replace(len(quantiles), np.nan)

        # Rewrite quantiles into integer, i.e. [.25, .5, .75, 1] as 4.
        if isinstance(quantiles, (list, tuple)):
            quantiles = len(quantiles)

        # Count the portfolio populations.
        portfolio_pop = PortfolioSort._count_portfolio_pop(quantile_sorts, quantiles)

        # Count the total number of assets per quantile for assertion checks later
        value, counter = np.unique(quantile_sorts, return_counts=True)
        asset_count = dict(zip(value, counter))

        if not all(asset_count.get(quant) for quant in range(quantiles)):
            raise ValueError("One or more portfolios has no assets ever assigned to.")

        # Determine valid time ticks by removing timestamps with less than X observations in each portfolio.
        valid_ticks = PortfolioSort.determine_valid_ticks(portfolio_pop, min_assets)
        valid_ticks = valid_ticks.reshape(-1)

        # Create empty dataframe for the results
        result = np.empty((valid_ticks.shape[0], quantiles))
        result.fill(np.nan)

        # Create leading returns in t=1, thus lagged characteristics and market value.
        lead_ret = np.roll(df_ret, char_lag, 0)

        # Calculate the weights and assign lagged weights to tomorrow's returns.
        max_mcap = np.nanquantile(df_mcap, q=value_weighted, axis=1)
        df_mcap = np.apply_along_axis(np.clip, 0, *[df_mcap, 0, max_mcap])

        # Calculate portfolio returns. Note, only correct for discrete returns!
        (low, high) = min(range(quantiles)), max(range(quantiles))
        for quantile in range(quantiles):

            if speedups and quantile not in (low, high):
                result[:, quantile] = np.NaN
                continue

            assets = np.nonzero(quantile_sorts == quantile)

            # Assert all asset are picked correctly
            assert all(quantile_sorts[assets] == quantile)
            assert len(assets[1]) == asset_count.get(quantile)

            mcap_quantile = np.empty(df_mcap.shape)
            mcap_quantile.fill(np.nan)
            mcap_quantile[assets] = df_mcap[assets].copy()

            if value_weighted:

                weights = np.divide(mcap_quantile, np.nansum(mcap_quantile, axis=1)[:, None])

                # Assert the portfolios weights sum up to 100%, tolerance level is 1%.
                (valid_weights, ) = np.nonzero(np.isclose(np.nansum(weights, axis=1), 1, rtol=0.01))
                valid_ticks = np.intersect1d(valid_ticks, valid_weights)

                # Calculate value-weighted portfolio returns
                result[:, quantile] = np.nansum(lead_ret[valid_ticks, :] *
                                                weights[valid_ticks, :],
                                                axis=1)

            else:
                # Calculate equally weighted returns. In order to not mess up the dimensions of the numpy array,
                # multiply the returns first with a boolean matrix of the shares and replace 0 with NaN.
                equal_weights = (~np.isnan(mcap_quantile[valid_ticks, :])).astype(float)
                equal_weights[equal_weights == 0] = np.nan

                result[:, quantile] = np.nanmean(lead_ret[valid_ticks, :] *
                                                 equal_weights,
                                                 axis=1)

        # Drop the last X observation(s), as these are calculated with returns from the beginning due to np.roll.
        result = result[: char_lag, :]

        # Calculate hedge-portfolio returns
        hedge_port = np.subtract(result[:, -1], result[:, 0])

        if speedups and get_series:
            timestamps = np.array(timestamps)[valid_ticks]
            return pd.DataFrame(data=hedge_port, index=timestamps[:char_lag], columns=["H-L"])

        result = np.concatenate([result, hedge_port[:, None]], axis=1)

        # Return the time series if specified. Note that the return series has to be shifted forwards by 'char_lag'
        # periods, as the returns where pulled back, whereas the market value and characteristic exposure remained.
        if get_series:
            timestamps = np.array(timestamps)[valid_ticks]
            series = pd.DataFrame(data=result, index=timestamps[:char_lag], columns=[*range(1, quantiles + 1), "H-L"])
            return series.shift(np.abs(char_lag))

        # Calculate t-statistic and assign either statistic or p-value
        t, p = np.apply_along_axis(tt, 0, result, **{"popmean": 0, "nan_policy": "omit"})

        # Return the time-series portfolio means and results from the t-test
        temp = pd.DataFrame({"Returns": np.nanmean(result, axis=0), "t-Test": t if get_tstat else p},
                            index=[*range(1, quantiles + 1), "H-L"])
        return temp

    @staticmethod
    def _get_valid_observations(sort_by: np.ndarray, returns: np.ndarray, market_cap: np.ndarray) -> np.ndarray:
        """ Return a logical mask with valid observations where characteristic, df_mcap and returns are
            all available."""

        valid = ~np.isnan(np.stack([sort_by, returns, market_cap], axis=2))

        return np.squeeze(np.all(valid, axis=2))

    @staticmethod
    @jit(nopython=True)
    def _assign_quantiles(data: np.array,
                          thresholds: Optional[np.ndarray] = None) -> np.array:
        """ Assigns the corresponding quantile to each asset"""

        return [np.searchsorted(thresholds[i, :], data[i, :], side="left") for i in range(data.shape[0])]

    @staticmethod
    @jit(nopython=True)
    def _get_asset_ids(sorted_values: np.ndarray, threshold_val: float, min_required: int = 50) -> np.ndarray:
        """ Returns the Asset Index fulfilling the threshold value.
            :param sorted_values: Sorted market cap values (descending)
            :param threshold_val: The threshold value to determine assets for the breakpoint calculation.
            :param min_required: Minimum required asset to calculate breakpoints of, default: 50.

            @return ToDO
        """
        result = np.empty((sorted_values.shape[0], 1))
        result.fill(np.nan)

        for i in range(0, sorted_values.shape[0]):
            cols = np.nonzero(sorted_values[i, :] <= threshold_val)
            cols = np.append(cols[0], min_required)
            result[i] = max(cols)

        return result

    @staticmethod
    @jit(nopython=True)
    def _get_threshold_breakpoints(x: np.ndarray, market_cap_sorted_index: np.ndarray,
                                   asset_cols: np.ndarray, quantiles: list) -> np.ndarray:
        # Create an empty array to save the breakpoints in
        breakpoints = np.empty((x.shape[0], len(quantiles)))
        breakpoints.fill(np.nan)

        for i in range(0, x.shape[0]):
            max_asset = asset_cols[i]
            data_array = x[i, :]
            all_asset_ids = market_cap_sorted_index[i, 0:max_asset]
            breakpoints[i, :] = np.nanquantile(data_array[all_asset_ids], quantiles)

        return breakpoints

    @staticmethod
    @jit(nopython=True)
    def _get_breakpoints(x: np.ndarray, quantiles: list):
        breakpoints = np.empty((x.shape[0], len(quantiles)))
        breakpoints.fill(np.nan)

        for i in range(0, x.shape[0]):
            breakpoints[i, :] = np.nanquantile(x[i, :], quantiles)

        return breakpoints

    @staticmethod
    @jit(nopython=True)
    def determine_valid_ticks(portfolio_pop: np.ndarray, min_assets: int) -> np.ndarray:
        """ Determine valid ticks which have the minimum required assets in all portfolios."""

        return np.argwhere([np.all(portfolio_pop[i, :] >= min_assets) for i in range(portfolio_pop.shape[0])])

    @staticmethod
    def _get_mutual_index(min_cs_observations: int, *args) -> tuple:
        """ Check for mutually shared columns and indexes."""
        args = list(args)

        # Iterate over inputs and drop all NaN rows and columns
        for i in range(len(args)):
            temp = args[i]
            temp.dropna(how="all", axis=0, inplace=True)
            temp.dropna(how="all", axis=1, inplace=True)
            args[i] = temp[(temp.notna().sum(axis=1) > min_cs_observations)].copy()

        common_timestamp = list(set.intersection(*map(set, [df.index for df in args])))
        common_timestamp.sort()

        common_columns = list(set.intersection(*map(set, [df.columns for df in args])))
        common_columns.sort()

        if len(common_columns) == 0 or len(common_timestamp) == 0:
            raise IndexError("No shared index found. Check input index types and values. \n"
                             "Shared indexes: %d - Shared columns: %d" % (len(common_timestamp), len(common_columns)))

        return common_timestamp, common_columns

    @staticmethod
    def _count_portfolio_pop(quantile_sorts: np.ndarray, quantiles: int) -> np.ndarray:
        """ Count the number of assets in each portfolio each period."""
        population = np.zeros((quantile_sorts.shape[0], len(np.unique(quantile_sorts))))

        for i in range(quantile_sorts.shape[0]):
            port, counter = np.unique(quantile_sorts[i, :], return_counts=True)
            population[i, port] = counter

        if PortfolioSort.has_nan and len(np.unique(quantile_sorts)) > quantiles:
            return population[:, :-1]
        else:
            return population
