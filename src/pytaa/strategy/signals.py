"""Computes signals used in the asset allocation schemes."""

import numpy as np
import pandas as pd


class Signal:
    """Momentum signal class used for tactical asset allocation."""

    def __init__(self, prices: pd.DataFrame):
        """Initialize signal class with daily returns.

        Args:
            prices (pd.DataFrame): table of daily returns
        """
        self.prices = prices
        self.monthly_prices = self.prices.resample("BME").last()

    def classic_momentum(self, start: int = 12, end: int = 1) -> pd.DataFrame:
        r"""Classic cross-sectional Momentum definition by Jegadeesh.

        The calculation follows:

        .. math::
            Z = \frac{P_{t-12}}{P_{t-1}} - 1

        For reference also see Asness (1994, 2013, 2014).

        Args:
            start (int, optional): beginning of momentum period. Defaults to 12.
            end (int, optional): end of momentum period. Defaults to 1.

        Returns:
            pd.DataFrame: table of momentum signal
        """
        momentum = self.monthly_prices.shift(end).div(self.monthly_prices.shift(start)) - 1
        return momentum

    def momentum_score(self) -> pd.DataFrame:
        """Calculate weighted average momentum for Vigilant portfolios."""
        score = np.zeros_like(self.monthly_prices)

        for horizon in [12, 4, 2, 1]:
            lag = int(12 / horizon)
            returns = self.monthly_prices.div(self.monthly_prices.shift(lag))
            score = score + (horizon * returns)

        norm_score = score - 19
        return norm_score

    def sharpe_ratio_score(self, duration: int, risk_free: float = 0.0, f: float = 1.0) -> pd.DataFrame:
        """Compute Sharpe ratio score for each asset at each monthly date.

        For each month (using `self.monthly_prices` index) this method looks back
        `duration` calendar days on the original daily `self.prices`, computes daily
        returns over that window, and then computes an annualized Sharpe ratio per
        asset:

            sharpe = (mean_daily_returns*252 - risk_free) / (std_daily_returns*sqrt(252))

        Args:
            duration (int): number of past days to use for Sharpe calculation.
            risk_free (float): annual risk-free rate (expressed in same units as returns,
                e.g. 0.0 for 0%). Defaults to 0.0.

        Returns:
            pd.DataFrame: DataFrame indexed by monthly dates with columns for each asset
                containing the Sharpe ratio (NaN where insufficient data).
        """
        # Prepare output DataFrame with same index and columns as monthly_prices
        sharpe_df = pd.DataFrame(index=self.monthly_prices.index, columns=self.monthly_prices.columns, dtype=float)

        # Annualization factors
        trading_days = 252.0
        sqrt_td = np.sqrt(trading_days)

        for date in self.monthly_prices.index:
            # define window: include days up to and including the monthly date
            end = date
            start = end - pd.Timedelta(days=duration)

            window = self.prices.loc[start:end]
            if window.shape[0] < 2:
                # not enough daily points to compute returns
                continue

            # compute daily returns for window
            daily_rets = window.pct_change().dropna(how="all")
            if daily_rets.shape[0] < 1:
                continue

            # mean and std per column
            mean_daily = daily_rets.mean(axis=0)
            std_daily = daily_rets.std(axis=0, ddof=1)

            # avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                sharpe = (mean_daily * trading_days - risk_free) / (pow(std_daily, f) * sqrt_td)

            # assign to output row
            sharpe_df.loc[date] = sharpe

        return sharpe_df


    def sma_crossover(
        self, lookback: int = 12, crossover: bool = True, days: int = 21
    ) -> pd.DataFrame:
        r"""Calculate simple Moving Average Crossover using monthly prices.

        Crossover score :math:`Z` over :math:`k` months is calculated as:

        .. math::

            Z = \frac{p_t}{k^{-1} \sum_{i=0}^{k} p_{t-i}} - 1

        If ``crossover`` is set to ``False`` then the simple moving average on its own is returned.

        Args:
            lookback (int, optional): number of months used for average. Defaults to 12.
            crossover (int, optional): returns crossover else just SMA. Defaults to True.
            days (int, optional): number of days in a business month. Defaults to 21.

        Returns:
            pd.DataFrame: table of signal strength
        """
        sma = self.prices.rolling(days * lookback).mean().resample("BME").last()
        if crossover:
            return self.monthly_prices.div(sma) - 1
        return sma

    def average_return(self) -> pd.DataFrame:
        """Calculate average return over the last 12 months."""
        avg_return = np.zeros_like(self.monthly_prices)
        for horizon in [1, 3, 6, 12]:
            avg_return += 0.25 * self.monthly_prices.div(self.monthly_prices.shift(horizon))
        return avg_return - 1

    def protective_momentum_score(self,risk_assets) -> pd.DataFrame:
        """Calculate momentum score for Generalized Protective Momentum portfolios."""
        returns = self.monthly_prices.pct_change()
        ew_basket = returns[risk_assets].mean(axis=1).rename("ew_basket")
        #ew_basket = returns.mean(axis=1).rename("ew_basket")
        grouper = returns.join(ew_basket)
        rolling_corr = grouper.rolling(12).corr(grouper["ew_basket"])

        avg_return = np.zeros_like(self.monthly_prices)
        for horizon in [1, 3, 6, 12]:
            avg_return += 0.25 * self.monthly_prices.div(self.monthly_prices.shift(horizon))

        score = (avg_return-1) * (1 - rolling_corr)
        return score