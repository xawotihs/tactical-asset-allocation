"""Store base functions for weights and positions."""

from datetime import datetime
from typing import Callable, List

import numpy as np
import pandas as pd

from pytaa.tools.risk import (
    Covariance,
    calculate_min_variance_portfolio,
    calculate_risk_parity_portfolio,
    calculate_rolling_volatility,
    weighted_covariance_matrix,
)


class Positions:
    """Base class for all portfolio weight classes."""

    def __init__(self, assets: List[str], rebalance_dates: List[datetime]):
        """Initialize base class with assets and rebalance dates.

        Creates dataframe ``self.weights`` which will be inherited downstream.

        Args:
            assets (List[str]): list of tickers
            rebalance_dates (List[str]): list of rebalance dates
        """
        self.assets = assets
        self.rebalance_dates = pd.DatetimeIndex(rebalance_dates, tz="utc").tz_convert(None)
        self.n_assets, self.n_obs = len(assets), len(self.rebalance_dates)

        # set up multilevel index
        index = pd.MultiIndex.from_product([self.rebalance_dates, assets])
        self.weights = pd.DataFrame(index=index)


class EqualWeights(Positions):
    """Store equally weighted portfolio."""

    def __init__(self, assets: List[str], rebalance_dates: List[datetime], name: str = "EW"):
        """Initialize and create equally weighted portfolio."""
        super().__init__(assets, rebalance_dates)
        self.__name__ = name
        position = np.ones(self.n_assets) / self.n_assets
        self.weights[self.__name__] = np.tile(position, self.n_obs)
        self.weights.sort_index(inplace=True)
        self.weights.index.names = ["Date", "ID"]


class StaticAllocation(Positions):
    """Fixed/static allocation across assets.

    Use this class to create a constant allocation across rebalance dates. The
    allocation can be provided as a dictionary mapping tickers to weights or as
    an iterable (list/ndarray) with the same order as the provided `assets`.

    Example:
        StaticAllocation(['SPY', 'TLT'], rebalance_dates, {'SPY': 0.6, 'TLT': 0.4}, name='60_40')
    """

    def __init__(self, assets: List[str], rebalance_dates: List[datetime], allocation, name: str = "STATIC"):
        """Create a static allocation.

        Args:
            assets (List[str]): list of tickers in the portfolio (order matters for list allocation).
            rebalance_dates (List[datetime]): list of rebalance dates.
            allocation (dict|list|np.ndarray): target weights either as dict {ticker: weight}
                or as list/array matching the order of `assets`.
            name (str): column name for the weights DataFrame.
        """
        super().__init__(assets, rebalance_dates)
        self.__name__ = name

        # build position vector in the order of self.assets
        if isinstance(allocation, dict):
            position = np.array([float(allocation.get(a, 0.0)) for a in self.assets], dtype=float)
        else:
            position = np.asarray(allocation, dtype=float)
            if position.ndim != 1 or position.shape[0] != self.n_assets:
                raise ValueError("Allocation must be a 1D array with the same length as assets")

        # normalize to sum to 1 if not already
        s = position.sum()
        if s == 0:
            raise ValueError("Allocation sums to zero")
        position = position / s

        self.weights[self.__name__] = np.tile(position, self.n_obs)
        self.weights.sort_index(inplace=True)
        self.weights.index.names = ["Date", "ID"]


class RiskParity(Positions):
    """Store naive implementation of Risk Parity."""

    def __init__(
        self,
        assets: List[str],
        rebalance_dates: List[datetime],
        returns: pd.DataFrame,
        estimator: str = "hist",
        lookback: int = 21,
        **kwargs: dict
    ):
        """Initialize class for Risk Parity calculation.

        Args:
            assets (List[str]): list of asset tickers
            rebalance_dates (List[datetime]): list of rebalancing dates
            returns (pd.DataFrame): dataframe of daily asset returns
            estimator (str, optional): volatility estimation method. Defaults to "hist".
            lookback (float, optional): volatility estimation window. Defaults to 21.
        """
        super().__init__(assets, rebalance_dates)
        self.returns = returns
        self.__name__ = "RP"

        # calculate historical vol (different estimators shall be used later)
        self.weights = self.create_weights(estimator, lookback, **kwargs)
        self.weights.index.names = ["Date", "ID"]

    def create_weights(self, method: str, lookback: float, **kwargs: dict) -> pd.DataFrame:
        """Create risk parity weights and store them in dataframe.

        The estimator can be one of: ``ewma``, ``hist``, ``equal_risk`` or ``min_variance``. The
        latter two involve an optimization process.

        Additional estimation parameters can be passed as keyword arguments. For example you can
        pass the halflife parameter for ``alpha=0.94`` when using ``ewma``.

        Args:
            method (str, optional): volatility estimation method.
            lookback (float, optional): volatility estimation window.
            **kwargs: keyword arguments for volatility estimation

        Returns:
            pd.DataFrame: weights for each asset
        """
        if method in ["hist", "ewma"]:
            inverse_vols = 1 / calculate_rolling_volatility(
                self.returns, estimator=method, lookback=lookback, **kwargs
            )
            inverse_vols = inverse_vols.reindex(self.rebalance_dates)
            weights = inverse_vols.div(inverse_vols.sum(axis=1).values.reshape(-1, 1))

        elif method in ["equal_risk", "min_variance"]:
            optimizer = {
                "min_variance": calculate_min_variance_portfolio,
                "equal_risk": calculate_risk_parity_portfolio,
            }
            weights = rolling_optimization(
                self.returns, self.rebalance_dates, optimizer[method], lookback, **kwargs
            )
        else:
            raise NotImplementedError

        weights = np.maximum(0, weights)
        return weights.stack().rename(self.__name__).to_frame()


def rolling_optimization(
    returns: pd.DataFrame,
    rebalance_dates: List[datetime],
    optimizer: Callable,
    lookback: int,
    shrinkage: str = None,
    shrinkage_factor: float = None,
) -> pd.DataFrame:
    """Perform rolling optimization over rebalance dates.

    Args:
        returns (pd.DataFrame): dataframe of daily asset returns
        rebalance_dates (List[datetime]): list of rebalancing dates
        optimizer (Callable): optimization routine, e.g. ``calculate_risk_parity_portfolio``
        lookback (int): window for covariance data
        shrinkage (str, optional): covariance shrinkage method. Defaults to None.
        shrinkage_factor (float, optional): covariance shrinkage factor. Defaults to None.

    Returns:
        pd.DataFrame: table of optimized asset weights
    """
    weights = []

    for date in rebalance_dates:
        data = returns[returns.index <= date].iloc[-lookback:, :]
        cov = Covariance(data, shrinkage, shrinkage_factor)

        w_opt = optimizer(cov)
        weights.append(w_opt)

    weights = pd.DataFrame(np.vstack(weights), index=rebalance_dates, columns=returns.columns)
    return weights


def vigilant_allocation(
    data: pd.Series,
    risk_assets: List[str],
    safe_assets: List[str],
    top_k: int = 5,
    step: float = 0.25,
) -> pd.DataFrame:
    """Allocate assets based on threshold using scores.

    Used in computing the Vigilant portfolios. The allocation works as follows (using $k=5$):
    Determine the number of assets $n$ with negative $Z$, if $n>4$ allocate 100% in safe asset with
    highest momentum score, if $n=3$ put 75% in safest asset, remaining 25% is split equally in 5
    risk assets with highest momentum, if $n=2$ put 50% in safest asset, 50% split evenly top 5
    risk assets etc.

    Args:
        data (pd.Series): dataframe with signals
        risk_assets (List[str]): list of risky assets
        safe_assets (List[str]): list of safety assets
        top_k (int, optional): rank threshold. Defaults to 5.
        step (float, optional): step in allocation to risk assets given signal. Defaults to 0.25.

    Returns:
        pd.DataFrame: dataframe of weights
    """
    is_neg = sum(np.where(data < 0, 1, 0))
    empty = data * np.nan
    safety = pd.concat([data.loc[safe_assets].rank(ascending=False), empty.loc[risk_assets]])
    safety = safety[~safety.index.duplicated()].sort_index()
    risky = pd.concat([data.loc[risk_assets].rank(ascending=False), empty.loc[safe_assets]])
    risky = risky[~risky.index.duplicated()].sort_index()

    # allocate assets based on number of negative scores
    safe_weights = np.where(safety == 1, min([1, step * is_neg]), 0)
    risk_weights = np.where(risky <= top_k, (1 - min([1, step * is_neg])) / top_k, 0)
    weights = safe_weights + risk_weights
    return pd.DataFrame(weights, index=safety.index).T


def top_n_equal_weights(signals: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Build equal-weighted top-N weights from a signal DataFrame.

    For each row (rebalance date) the function selects up to ``top_n`` assets
    with the largest positive signal values and assigns them equal weights
    summing to 1. If fewer than ``top_n`` assets have positive signals the
    positive assets receive equal weight. If no assets have positive signals
    the row is all zeros.

    Args:
        signals (pd.DataFrame): index = dates, columns = asset identifiers; may
            contain NaN values. NaNs are treated as non-positive.
        top_n (int): number of top assets to select per row. Must be >= 1.

    Returns:
        pd.DataFrame: weights in the same wide format as ``signals`` with
        float weights (rows sum to 1 when any positive signals exist).
    """
    if not isinstance(signals, pd.DataFrame):
        raise TypeError("signals must be a pandas DataFrame")
    if not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be an integer >= 1")

    # Treat NaN as non-positive so they are not selected
    sig = signals.fillna(-np.inf)

    def _row_top_weights(row: pd.Series) -> pd.Series:
        # consider only positive signals
        positives = row[row > 0.0]
        if positives.empty:
            return pd.Series(0.0, index=row.index)

        n = min(top_n, positives.shape[0])
        top_idx = positives.nlargest(n).index
        w = pd.Series(0.0, index=row.index)
        w.loc[top_idx] = 1.0 / float(n)
        return w

    weights = sig.apply(_row_top_weights, axis=1)
    # preserve any index/column names from input
    weights.index.name = signals.index.name
    weights.columns.name = signals.columns.name
    return weights


def best_mix_between_weights(
    weights_list: List[pd.DataFrame],
    alloc_ranges: List[tuple],
    prices: pd.DataFrame,
    lookback_days: int = 21,
    step_pct: int = 10,
) -> tuple:
    """Choose best allocation per rebalance date between multiple weight tables.

    This function accepts a list of wide-format weight DataFrames (index=dates,
    columns=assets) and a matching list of allocation ranges for each weights
    DataFrame. Each allocation range is a tuple (min_pct, max_pct) expressed in
    percent (0-100). The function enumerates all allocations that are multiples
    of ``step_pct`` within each range and whose sum is 100, evaluates the
    portfolio Sharpe over the prior ``lookback_days`` of daily returns, and
    selects the allocation with the highest Sharpe for each rebalance date.

    Args:
        weights_list: list of pd.DataFrame weight tables (wide format)
        alloc_ranges: list of (min_pct, max_pct) tuples matching weights_list
        prices: DataFrame of asset prices with DatetimeIndex
        lookback_days: lookback window to compute Sharpe (trading days)
        step_pct: allocation granularity in percent (default 10)

    Returns:
        combined_weights (pd.DataFrame): combined wide-format weights per date
        summary (pd.DataFrame): per-date summary with chosen allocation and Sharpe
    """
    from itertools import product

    # basic validation
    if not isinstance(weights_list, list) or len(weights_list) == 0:
        raise TypeError("weights_list must be a non-empty list of DataFrames")
    if not isinstance(alloc_ranges, list) or len(alloc_ranges) != len(weights_list):
        raise TypeError("alloc_ranges must be a list of the same length as weights_list")
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame of asset prices")

    # union of assets and rebalance dates
    # normalize weights_list: accept either wide-format (index=dates, columns=assets)
    # or stacked MultiIndex (Date, ID) where values are weights (common in this repo).
    norm_weights = []
    for w in weights_list:
        # make a copy to avoid mutating caller
        ww = w.copy()
        # if stacked (MultiIndex) convert to wide by unstacking the second level
        if getattr(ww.index, "nlevels", 1) == 2:
            # If DataFrame with multiple columns, try to collapse by unstacking each column
            if ww.shape[1] == 1:
                s = ww.iloc[:, 0]
                wide = s.unstack(level=1)
            else:
                # when multiple columns, prefer to take each column separately and sum across
                # columns after unstacking to produce a single wide table per input
                unstacked_list = [ww.iloc[:, i].unstack(level=1) for i in range(ww.shape[1])]
                # align and sum (treating NaN as 0)
                wide = pd.concat(unstacked_list, axis=0).groupby(level=0).sum()
        else:
            wide = ww
        # ensure index is DatetimeIndex (no tz) for comparisons
        if not isinstance(wide.index, pd.DatetimeIndex):
            try:
                wide.index = pd.DatetimeIndex(wide.index, tz="utc").tz_convert(None)
            except Exception:
                # fallback: coerce via to_datetime
                wide.index = pd.to_datetime(wide.index)
        norm_weights.append(wide)

    # replace weights_list with normalized wide tables
    weights_list = norm_weights

    cols = sorted({c for w in weights_list for c in w.columns.astype(str)}.union(set(prices.columns.astype(str))))
    dates = sorted({d for w in weights_list for d in w.index})

    # precompute price returns
    price_rets = prices.reindex(columns=cols).pct_change()

    # prepare allowed allocation choices per weight (in percent)
    alloc_choices = []
    for (mn, mx) in alloc_ranges:
        if not (0 <= mn <= mx <= 100):
            raise ValueError("allocation ranges must be between 0 and 100 and min <= max")
        choices = list(range(int(mn), int(mx) + 1, int(step_pct)))
        if len(choices) == 0:
            raise ValueError("no allocation choices generated for a range; adjust step_pct or range")
        alloc_choices.append(choices)

    # helper: generate candidate allocations that sum to 100
    candidate_allocs = []
    for comb in product(*alloc_choices):
        if sum(comb) == 100:
            candidate_allocs.append(list(comb))

    if len(candidate_allocs) == 0:
        raise ValueError("no candidate allocations sum to 100 with the given ranges and step_pct")

    combined_rows = []
    summary_records = []

    # helper to produce a default allocation when not enough history: allocate mins then distribute
    def _default_alloc(ranges, step):
        alloc = [int(r[0]) for r in ranges]
        remaining = 100 - sum(alloc)
        if remaining < 0:
            raise ValueError("sum of minimums in ranges exceeds 100")
        # distribute remaining in step increments to those who can accept more
        i = 0
        n = len(alloc)
        while remaining > 0:
            idx = i % n
            max_add = int(ranges[idx][1]) - alloc[idx]
            add = min(step, max_add, remaining)
            if add > 0:
                alloc[idx] += add
                remaining -= add
            i += 1
            # guard: if loop cycles without progress, break
            if i > n * 100:
                break
        if remaining != 0:
            # as fallback normalize to nearest step
            base = [int(round(a / step) * step) for a in alloc]
            total = sum(base)
            if total == 0:
                raise ValueError("cannot build default allocation from ranges")
            factor = 100 / total
            base = [int(round(b * factor / step) * step) for b in base]
            # adjust last
            diff = 100 - sum(base)
            base[-1] += diff
            return base
        return alloc

    # evaluate each date
    for date in dates:
        # build group weight rows for this date
        rows = [w.reindex(index=[date], columns=cols).fillna(0.0).iloc[0] for w in weights_list]

        # find end_date for returns (last available <= date)
        available_dates = price_rets.index[price_rets.index <= date]
        if len(available_dates) == 0:
            # skip dates with no price history
            continue
        end_date = available_dates.max()

        rets_window = price_rets.loc[:end_date].iloc[-lookback_days:]
        if rets_window.empty or rets_window.shape[0] < 2:
            # not enough history: pick default allocation
            chosen_alloc = _default_alloc(alloc_ranges, step_pct)
            sharpe_map = {tuple(a): None for a in candidate_allocs}
        else:
            # compute each group's return series
            group_rets = []
            for r in rows:
                grp = (rets_window * r).sum(axis=1)
                group_rets.append(grp)

            # evaluate candidates
            best_sharpe = None
            chosen_alloc = None
            sharpe_map = {}
            for alloc in candidate_allocs:
                # compute portfolio series
                port = sum((alloc[i] / 100.0) * group_rets[i] for i in range(len(alloc)))
                if port.std() == 0 or port.empty:
                    sharpe = None
                else:
                    sharpe = float((port.mean() * 252) / (port.std() * (252 ** 0.5)))
                sharpe_map[tuple(alloc)] = sharpe
                if sharpe is not None and (best_sharpe is None or sharpe > best_sharpe):
                    best_sharpe = sharpe
                    chosen_alloc = alloc

            if chosen_alloc is None:
                # fallback to default if no valid sharpe found
                chosen_alloc = _default_alloc(alloc_ranges, step_pct)

        # build combined weight row
        combined = sum((chosen_alloc[i] / 100.0) * rows[i] for i in range(len(rows)))
        combined.name = date
        combined_rows.append(combined)

        # record summary
        rec = {"date": date, "chosen_alloc": tuple(chosen_alloc), "chosen_sharpe": sharpe_map.get(tuple(chosen_alloc))}
        # optionally include top few candidate sharpes (store as stringified dict)
        rec["candidates_tried"] = len(candidate_allocs)
        summary_records.append(rec)

    if len(combined_rows) == 0:
        return pd.DataFrame(columns=cols), pd.DataFrame()

    combined_weights = pd.DataFrame(combined_rows).sort_index()
    summary = pd.DataFrame(summary_records).set_index("date").sort_index()
    return combined_weights, summary


def kipnis_allocation(
    returns: pd.DataFrame,
    signals: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    canary_assets: List[str],
    risk_assets: List[str],
    safe_assets: List[str],
    top_k: int = 5,
) -> pd.DataFrame:
    """Kipnis defensive asset allocation scheme.

    The strategy allocates based on a momentum score, which is a weighted average over different
    windows. A canary universe is used to allocate either towards risk or safe assets, e.g. if all
    canary assets are positively trending, then allocate 100% towards the investment universe. If
    only one canary asset (out of two) has positive trend, then put 50% into crash protection (e.g.
    a govy ETF like ``IEF``). If no assets has positive trend, then pull the entire funds into crash
    protection, that is a cash ETF like ``BIL``.

    The final allocation is done using the minimum variance portfolio with a cleaned covariance
    matrix. It uses a weighted average correlation (over 4 windows) and the 1 month realised
    volatility.

    The strategy is more closely explained in:
    https://quantstrattrader.com/2019/01/24/right-now-its-kda-asset-allocation/

    Args:
        returns (pd.DataFrame): table of asset returns
        signals (pd.DataFrame): table of asset signals
        rebalance_dates (List[pd.Timestamp]): list of rebalance dates
        canary_assets (List[str]): list of canary or proxy assets
        risk_assets (List[str]): list of risky assets (e.g. equities)
        safe_assets (List[str]): list of safe assets (e.g. govies)
        top_k (int, optional): number of risky assets to include. Defaults to 5.

    Returns:
        pd.DataFrame: table of returns
    """
    all_weights = []
    rebal_dates = [i for i in rebalance_dates if i >= signals.index[0]]

    for date in rebal_dates:
        signal_sample = signals[signals.index <= date].iloc[-1, :]
        risk_on = set(signal_sample[signal_sample > 0].index).intersection(canary_assets)
        ranked = signal_sample[risk_assets].rank(ascending=False)
        buy_assets = ranked[ranked <= top_k].index

        # if canary assets are positive then go risk on, else allocate to safe assets
        if len(risk_on) == len(canary_assets):
            canary_weight = 0
        elif len(risk_on) == 0:
            equal_weight = np.ones((1, len(safe_assets))) / len(safe_assets)
            weights = pd.DataFrame(equal_weight, columns=safe_assets, index=[date])
            all_weights.append(weights)
            continue
        elif signal_sample[safe_assets].le(0).any():
            weights = pd.DataFrame({"BIL": [1]}, index=[date])
            all_weights.append(weights)
            continue
        else:
            canary_weight = 0.5

        # calculate min variance portfolio for top k risk assets
        return_sample = returns.loc[returns.index <= date, list(buy_assets)].iloc[-260:, :]
        w_cov = weighted_covariance_matrix(return_sample)
        risky_weights = calculate_min_variance_portfolio(w_cov).reshape(1, -1)

        # stack risk and safe assets together, weighting them by the canary factor
        safe_weights = np.ones((1, len(safe_assets))) / len(safe_assets)
        weights = np.hstack([(1 - canary_weight) * risky_weights, canary_weight * safe_weights])
        cols = list(buy_assets) + list(safe_assets)
        weights = pd.DataFrame(weights, index=[date], columns=cols)

        # if any assets show up twice remove them
        weights = weights.loc[:, ~weights.columns.duplicated()]
        all_weights.append(weights)
    return pd.concat(all_weights, join="outer").fillna(0)


def aqr_trend_allocation(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    risk_assets: List[str],
    cash_asset: str,
) -> pd.DataFrame:
    """Allocate portfolio to trending assets using AQR Risk Parity methodology.

    Puts share of non-trending assets into cash, depending on signal strength.ary_

    Args:
        returns (pd.DataFrame): table of asset returns
        signal (pd.DataFrame): table of signals, i.e. SMA crossovers
        rebalance_dates (pd.DatetimeIndex): list of rebalance dates
        risk_assets (List[str]): list of risky assets
        cash_asset (str): safe or cash asset

    Returns:
        pd.DataFrame: a table of portfolio weights
    """
    strategy_weights = []

    for date in rebalance_dates[rebalance_dates >= signal.dropna().index.min()]:
        return_sample = returns.loc[returns.index <= date].iloc[-260 * 3:, :]
        monthly_ret = return_sample.resample("BME").apply(lambda x: np.prod(1 + x) - 1)
        excess_ret = monthly_ret[risk_assets].sub(monthly_ret[[cash_asset]].values)

        # weight assets by inverse of vol: s_i = 1 / vol_i / (1 / sum[vol_i])
        inv_vol = 1 / excess_ret.std() * np.sqrt(12)
        buy_assets = signal.loc[signal.index <= date, risk_assets].iloc[-1].ge(0)
        buy_assets = buy_assets.index[buy_assets]

        # portfolio weights calculation
        risk_allocation = len(buy_assets) / len(risk_assets)
        risk_weights = inv_vol.loc[buy_assets] / inv_vol.loc[buy_assets].sum() * risk_allocation
        weights = risk_weights.rename(date).to_frame().T
        weights.columns.name, weights.index.name = "ID", "Date"
        weights[[cash_asset]] = 1 - risk_allocation
        strategy_weights.append(weights)

    return pd.concat(strategy_weights).fillna(0)

def protective_allocation(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    risk_assets: List[str],
    safe_assets: List[str],
    crash_protection: int = 2,
    top_k: int = 3,
) -> pd.DataFrame:
    """Allocate portfolio to trending assets using Generalized Protective Momentum.

    Args:
        returns (pd.DataFrame): table of asset returns
        signal (pd.DataFrame): table of signals, i.e. SMA crossovers
        rebalance_dates (pd.DatetimeIndex): list of rebalance dates
        risk_assets (List[str]): list of risky assets
        safe_assets (str): safe or cash asset
        crash_protection: how much crash protection to apply (0-4),
        top_k: how many risky assets to include

    Returns:
        pd.DataFrame: a table of portfolio weights
    """
    strategy_weights = []
    
    bf = signal[risk_assets].lt(0).sum(axis=1) / (len(risk_assets) - crash_protection * (len(risk_assets)/4))
    # Handle case where all values might be NA by filling NA with -inf
    cash_asset = signal[safe_assets].fillna(-float('inf')).idxmax(axis=1)

    rebal_dates = [i for i in rebalance_dates if (i >= signal.index[0] and i <= bf.dropna().index.max())]

    # for all dates in bf
    for date in rebal_dates:
        weights = None
        # if bf > 100% then allocate 100% to safe asset with highest momentum
        if bf.loc[date] > 1:
            weights = pd.DataFrame({cash_asset.loc[date]: [1]}, index=[date])
        # if 0 <= bf <= 100% then allocate bf% to safe asset with highest momentum
        else: #if bf.loc[date] > 0:
            if bf.loc[date] > 0:
                weights = pd.DataFrame({cash_asset.loc[date]: [bf.loc[date]]}, index=[date])
            else:
                weights = pd.DataFrame()
            # then allocate the remaining weight to the top k risk assets

            remaining_weight = 1 - bf.loc[date]
            top_k_assets = signal[risk_assets].loc[date].nlargest(top_k).index
            weights = pd.concat([weights, pd.DataFrame({a: [remaining_weight / top_k] for a in top_k_assets}, index=[date])], axis=1)
        #print(weights)
        strategy_weights.append(weights)
    return pd.concat(strategy_weights).fillna(0)

def haa_allocation(
    signal: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    risk_assets: List[str],
    safe_assets: List[str],
    canary_asset: str,
) -> pd.DataFrame:
    """Allocate portfolio to trending assets using hybrid asset allocation.

    Args:
        signal (pd.DataFrame): table of signals, i.e. SMA crossovers
        rebalance_dates (pd.DatetimeIndex): list of rebalance dates
        risk_assets (List[str]): list of risky assets
        safe_assets (List[str]): list of safe assets
        canary_asset (str): canary asset to determine risk-on or risk-off

    Returns:
        pd.DataFrame: a table of portfolio weights
    """  
    strategy_weights = []
    
    # for all dates: build a boolean mask instead of using Python 'and' on arrays
    sd = signal.dropna()
    if sd.empty:
        # nothing to allocate when there are no valid signals
        return pd.DataFrame()

    mask = (rebalance_dates >= sd.index.min()) & (rebalance_dates <= sd.index.max())
    for date in rebalance_dates[mask]:
        # select the cash asset as the asset in safe_asset list with the highest signal
        cash_asset = signal[safe_assets].loc[date].idxmax()

        # if the signal of the canary asset is negative, allocate 100% to the safe asset with the highest signal
        if signal.loc[date, canary_asset] <= 0:
            weights = pd.DataFrame({cash_asset: [1]}, index=[date])
        else:
            # for the top 4 risk assets, allocate 25% to each if the signal is positive
            top_k_assets = signal[risk_assets].loc[date].nlargest(4).index
            
            if len(top_k_assets) == 0:
                weights = pd.DataFrame({cash_asset: [1]}, index=[date])
            else:
                # Initialize weights dictionary with cash asset to prevent empty dict
                weight_dict = {cash_asset: 0}
                asset_weight = 1.0 / len(top_k_assets)  # Equal weight for each of the top 4 assets
                
                # for each asset in top_k_assets, allocate equal weight if signal is positive, 
                # otherwise allocate that weight to the cash asset
                for asset in top_k_assets:
                    if signal.loc[date, asset] > 0:
                        weight_dict[asset] = asset_weight
                    else:
                        weight_dict[cash_asset] += asset_weight
                
                weights = pd.DataFrame(weight_dict, index=[date])
            
        strategy_weights.append(weights)

    return pd.concat(strategy_weights).fillna(0)
