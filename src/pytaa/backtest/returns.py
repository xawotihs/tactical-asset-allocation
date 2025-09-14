"""Calculate portfolio strategy returns."""

import numba
import numpy as np
import pandas as pd

from pytaa.tools.data import (
    get_currency_returns,
    get_historical_dividends,
    get_historical_price_data,
    get_issue_currency_for_tickers,
)

pd.options.mode.chained_assignment = None


def get_historical_total_return(
    price_data: pd.DataFrame, portfolio_currency: str = None, return_type: str = "total"
) -> pd.DataFrame:
    r"""Calculate daily total return in portfolio currency.

    The one day total return :math:`r_t` is calculated as:

    .. math::

        r_{t,t-1}=\frac{p_t + d_t - p_{t-1}}{p_{t-1}}

    The dividends :math:`d_t` are retrieved from Yahoo! Finance. We use the close price instead
    of the adjusted close price. Both are stock split adjusted but the adjusted close is also
    backwards-adjusted for dividends, i.e. the dividend paid is subtract from the denominator in
    the return calculation.

    Args:
        price_data (pd.DataFrame): table of closing price, are already converted to USD and adjusted for splits.
        portfolio_currency (str, optional): portfolio currency for returns. Defaults to None.
        return_type (str, optional): returns gross or net of dividends. Defaults to "total".

    Returns:
        pd.DataFrame: table of asset returns

    Raises:
        NotImplementedError: if return_type is not equal to "price"
    """
    start_date, end_date = price_data.index.min(), price_data.index.max()
    tickers = price_data.columns.to_list()

    if return_type == "price":
        returns = price_data.pct_change().dropna()
    elif return_type == "total":
        dividends = get_historical_dividends(tickers, start_date, end_date)
        dividends = dividends.reindex(price_data.index).fillna(0).loc[:, tickers]
        returns = price_data.add(dividends).div(price_data.shift(1).values).sub(1).dropna()
    else:
        raise NotImplementedError

    if portfolio_currency is None:
        # no conversion required
        return returns

    # get fx returns for each ticker currency to the desired portfolio currency
    fx_returns = get_currency_returns(['USD'], start_date, end_date, portfolio_currency)
    fx_index = (1 + fx_returns).cumprod()

    # adjust price series for tickers whose issue currency != portfolio_currency
    for ticker in returns.columns:
        # multiply the USD price series by the FX index to get price in portfolio currency
        price_data[ticker] = price_data[ticker].multiply(fx_index['USD'], axis=0)

    # recompute returns on the converted prices
    if return_type == "price":
        returns = price_data.pct_change().dropna()
    else:
        dividends = get_historical_dividends(tickers, start_date, end_date)
        dividends = dividends.reindex(price_data.index).fillna(0).loc[:, tickers]
        returns = price_data.add(dividends).div(price_data.shift(1).values).sub(1).dropna()

    return returns


@numba.njit
def calculate_drifted_weight_returns(
    returns: np.array, weights: np.array, rebal_index: np.array
) -> np.array:
    r"""Project cumulative daily returns onto lower frequency returns.

    The portfolio weights are iteratively updated using market performance:

    .. math::

        w_{i,t} = \frac{w_{i,t-1} \times (1 + r_{i,t})}{\sum_j^N{w_{j,t-1} \times (1 + r_{j,t})}}\\
        r_{p,t} = \sum_i^N w_{i,t} \times r_{i,t}\\

    Args:
        weights (np.array): KxM table of portfolio weights
        returns (np.array): NxM table of daily returns
        rebal_index (np.array): Kx0 1d array of rebalance indices for returns
    Returns:
        np.array: Nx1 vector of drifted returns
    """
    n_obs, n_assets = returns.shape[0], weights.shape[1]
    total_return = np.zeros((n_obs, 1), dtype=np.float64)

    # initialize drift weights to the first provided weights row if available
    if weights.shape[0] > 0:
        w_drift = weights[0, :].astype(np.float64).copy()
    else:
        w_drift = np.zeros(n_assets, dtype=np.float64)

    # Iterate days: compute the day's portfolio return using the current (drifted)
    # weights, then if this day is a rebalance date, update w_drift so the new
    # weights apply from the next day (next-day semantics).
    for i in range(n_obs):
        r_day = returns[i, :]
        # portfolio return for the day uses current weights
        total_return[i, :] = np.nansum(w_drift * r_day)

        # normalized update of w_drift based on market moves
        denom = np.nansum(w_drift * (1 + r_day))
        if denom == 0:
            # avoid division by zero; keep current weights
            w_drift = w_drift
        else:
            w_drift = (w_drift * (1 + r_day)) / denom

        # if today is a rebalance date, replace w_drift with the target weights
        # so they will be used from the next day onwards
        if i in rebal_index:
            w_drift = weights[np.where(rebal_index == i)[0][0], :].astype(np.float64)

    return total_return


class Backtester:
    """Backtest strategies and calculate total returns given input weights.

    The input table should be multi-level where the first level points to the rebalancing date and
    the second level to the ticker of the asset. For example, a table might look like this:

    | Date       | ID   | Strategy Name |
    |------------|------|---------------|
    | 2020-01-31 | SPY  | 0.5           |
    | 2020-01-31 | AGG  | 0.5           |
    | 2020-02-28 | SPY  | 0.6           |
    | 2020-02-28 | AGG  | 0.4           |
    | 2020-03-31 | SPY  | 0.7           |
    | 2020-03-31 | AGG  | 0.3           |

    Note that the index is set to ``["Date", "ID"]`` in this case. The weights in the third column
    should add up to unity for each date.
    """

    def __init__(self, weights: pd.DataFrame, portfolio_currency: str = "USD", **kwargs):
        """Initialize backtester with weights matrix.

        Args:
            weights (pd.DataFrame): multi-level dataframe with weights
            portfolio_currency (str): currency in which returns are calculated. Defaults to USD.
        """
        self.weights = weights
        self.portfolio_currency = portfolio_currency
        self.rebal_dates = self.weights.index.get_level_values(0).unique()
        self.assets = self.weights.index.get_level_values(1).unique()
        self.strategies = self.weights.columns
        self.frequency = pd.infer_freq(self.rebal_dates) if len(self.rebal_dates) > 3 else None

    def _process_strategy(self, returns: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Project weights onto returns for a given strategy.

        Args:
            returns (pd.DataFrame): table of asset returns
            strategy (str): label of strategy

        Returns:
            pd.DataFrame: table of simple returns per strategy
        """
        strat_weights = self.weights[strategy].unstack().fillna(0)
        # Filter out future weight dates that are beyond our returns data
        strat_weights = strat_weights[strat_weights.index <= returns.index.max()]

        # select returns for the assets in the weight table
        strat_returns = returns.loc[:, strat_weights.columns]

        # include returns prior to the first rebalance: start from the returns series
        temp_idx = pd.bdate_range(returns.index.min(), strat_returns.index.max())
        strat_returns = strat_returns.reindex(temp_idx).fillna(0)

        # drop weight dates earlier than available returns (avoid get_loc KeyError)
        strat_weights = strat_weights[strat_weights.index >= strat_returns.index.min()]

        # map rebalancing dates into the reindexed returns index using get_indexer
        # this allows nearest/ffill mapping and avoids KeyError on exact lookup
        rebal_pos = strat_returns.index.get_indexer(strat_weights.index, method="pad")
        # if pad cannot find a position (returns -1), set to first available day
        rebal_pos = np.where(rebal_pos == -1, 0, rebal_pos)
        rebal_index = rebal_pos.tolist()

        total_return = calculate_drifted_weight_returns(
            np.asarray(strat_returns), np.asarray(strat_weights), np.asarray(rebal_index)
        )

        return pd.DataFrame(total_return, index=strat_returns.index, columns=[strategy])

    def run(
        self, end_date: str = None, frequency: str = None, return_type: str = "total", prices = None
    ) -> pd.DataFrame:
        """Run backtester and return strategy returns.

        Args:
            end_date (str, optional): end date of strategy. Defaults to None.
            frequency (str, optional): frequency of portfolio returns. Defaults to None.
            return_type (str, optional): either ``price`` or ``total`` return
        Returns:
            pd.DataFrame: table of portfolio returns

        Raises:
            NotImplementedError: if frequency is neither None or "D"
        """
        start_date = self.rebal_dates.min() - pd.offsets.BDay(1)
        if end_date is None:
            end_date = self.rebal_dates.max() + pd.offsets.BDay(1)

        # retrieve data for total return calculation
        if prices is None:
            prices = get_historical_price_data(self.assets, start_date, end_date, adjust=True, convert_to_usd=True).loc[:, "Close"]
        returns = get_historical_total_return(prices, self.portfolio_currency, return_type)

        portfolio_total_return = []

        for strategy in self.weights.columns:
            total_return = self._process_strategy(returns, strategy)
            portfolio_total_return.append(total_return)

        # concat all strategy returns, then aggregate to chosen frequency
        portfolio_total_return = pd.concat(portfolio_total_return, axis=1)
        portfolio_total_return.index.name = "Date"

        if frequency is None:
            resampled = portfolio_total_return.groupby(pd.Grouper(freq=self.frequency))
            return resampled.apply(lambda x: (1 + x).prod() - 1)
        if frequency == "D":
            return portfolio_total_return
        raise NotImplementedError
