"""Data handling and webscraping tools."""

import concurrent.futures
import logging
from datetime import datetime
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from lxml import etree, html

from pytaa.strategy.static import VALID_CURRENCIES
from pytaa.strategy.strategies import StrategyPipeline
from pytaa.tools.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def get_historical_dividends(
    tickers: List[str], start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    """Retrieve historical dividends for universe of stocks using Yahoo! Finance.

    Args:
        tickers (List[str]): list of equity tickers
        start_date (datetime): start date of series
        end_date (datetime): end date of series

    Returns:
        pd.DataFrame: table of dividends
    """
    dividends = [yf.Ticker(x).dividends.rename(x).to_frame() for x in tickers]
    div_table = pd.concat(dividends, axis=1)
    div_table.index = div_table.index.tz_localize(None)
    return div_table.loc[(div_table.index >= start_date) & (div_table.index <= end_date)]


def validate_tickers(tickers: List[str], raise_issue: bool = False) -> bool:
    """Validate list of security tickers.

    Args:
        tickers (List[str]): list of tickers to validate
        raise_issue (bool, optional): raises error if set to true. Defaults to False.

    Returns:
        bool: False if validation failed and else True

    Raises:
        Exception: if tickers are invalid and ``raise_issue`` has been set to True
    """
    is_valid = True
    if len(tickers) < 1 or any(x is None for x in tickers):
        is_valid = False

    if not is_valid and not raise_issue:
        return is_valid
    if not is_valid and raise_issue:
        raise Exception("Tickers are not valid in Yahoo! Finace. Review inputs before proceeding.")
    return True


def get_historical_price_data(
    tickers: List[str], 
    start_date: str, 
    end_date: str, 
    adjust: bool = False, 
    convert_to_usd: bool = False, 
    **kwargs: dict
) -> pd.DataFrame:
    """Request price data using Yahoo! Finance API.

    Additional keyword arguments can be passed to control if and how tickers are validated. You
    can pass ``raise_issue=True`` to raise an Exception if an invalid ticker is present.

    Args:
        tickers (List[str]): list of tickers
        start_date (str): start date of time series
        end_date (str): end date of time series
        adjust (bool): adjust yahoo-finance request for corporate actions
        convert_to_usd (bool): convert all non-USD prices to USD using historical exchange rates
        **kwargs (dict): key word arguments for ticker validation

    Returns:
        pd.DataFrame: table of prices
    """
    # Convert dates to string format
    start_date_str = start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d")
    end_date_str = end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d")
    
    # Validate and deduplicate tickers
    tickers = list(set(tickers))
    if not validate_tickers(tickers, raise_issue=bool(kwargs.get('raise_issue', False))):
        return pd.DataFrame(columns=["Close", "Open", "Low", "High", "Volume"])

    # Get price data
    data = yf.download(
        tickers, start=start_date_str, end=end_date_str, 
        auto_adjust=adjust, progress=False
    )
    
    if not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame(columns=["Close", "Open", "Low", "High", "Volume"])
    
    # Handle currency conversion if needed
    if convert_to_usd:
        # Get currency info for all tickers
        ticker_currencies = {
            ticker: find_ticker_currency(ticker, fallback="USD") 
            for ticker in tickers
        }
        
        # Get unique non-USD currencies
        currencies_to_convert = {
            curr for curr in ticker_currencies.values() 
            if curr != "USD"
        }
        
        if currencies_to_convert:
           
            # Convert each ticker's prices if needed
            if isinstance(data, pd.DataFrame) and not data.empty:
                for ticker in tickers:
                    currency = ticker_currencies[ticker]
                    if currency == "USD":
                        continue
                        
                    # Try direct rate first (e.g., CHFUSD=X)
                    fx_pair = f"{currency}USD=X"
                    fx_data = yf.download(fx_pair, start=start_date_str, end=end_date_str, auto_adjust=adjust, progress=False)
                    
                    # If direct rate not available, try inverse rate (e.g., USDCHF=X)
                    if not isinstance(fx_data, pd.DataFrame) or fx_data.empty:
                        fx_pair = f"USD{currency}=X"
                        fx_data = yf.download(fx_pair, start=start_date_str, end=end_date_str, auto_adjust=adjust, progress=False)
                        if isinstance(fx_data, pd.DataFrame) and not fx_data.empty:
                            # Invert the rate since we got USDXXX instead of XXXUSD
                            fx_data['Close'] = 1.0 / fx_data['Close']
                    
                    if isinstance(fx_data, pd.DataFrame) and not fx_data.empty:
                        # Build a numeric FX rate series aligned to the data index
                        fx_rates = fx_data['Close'].reindex(data.index)
                        if isinstance(fx_rates, pd.DataFrame) and fx_rates.shape[1] == 1:
                            fx_rates = fx_rates.iloc[:, 0]
                        fx_rates = pd.to_numeric(fx_rates, errors='coerce').ffill().replace(0, pd.NA).fillna(1.0)

                        # Handle single-level columns easily
                        if data.columns.nlevels == 1:
                            for col in ["Open", "High", "Low", "Close"]:
                                if col in data.columns:
                                    data[col] = pd.to_numeric(data[col], errors='coerce').mul(fx_rates, axis=0)
                        else:
                            # Identify which level contains the attributes (Open/Close/...)
                            lvl0 = list(data.columns.levels[0]) if hasattr(data.columns, 'levels') else []
                            lvl1 = list(data.columns.levels[1]) if hasattr(data.columns, 'levels') else []
                            if 'Close' in lvl0:
                                attr_level = 0
                            elif 'Close' in lvl1:
                                attr_level = 1
                            else:
                                # Cannot find Close level, skip this ticker
                                continue

                            # Extract Close prices as DataFrame with tickers as columns
                            close_prices = data.xs('Close', axis=1, level=attr_level)

                            # If close_prices columns are not strings (e.g., tuples), try to normalize
                            close_cols = list(close_prices.columns)

                            # Multiply only the relevant ticker column if present
                            if ticker in close_prices.columns:
                                col_series = pd.to_numeric(close_prices[ticker], errors='coerce')
                                if col_series.notna().any():
                                    converted = col_series.mul(fx_rates, axis=0)
                                    # Update the close_prices DataFrame with converted values
                                    close_prices = close_prices.copy()
                                    close_prices[ticker] = converted
                                    # Now put close_prices back into the original DataFrame safely
                                    # Build a DataFrame with the same MultiIndex columns as the Close slice
                                    if attr_level == 0:
                                        # close_prices columns should be under ('Close', ticker)
                                        # create a column MultiIndex matching data for the Close level
                                        new_cols = pd.MultiIndex.from_tuples([('Close', c) for c in close_prices.columns])
                                    else:
                                        new_cols = pd.MultiIndex.from_tuples([(c, 'Close') for c in close_prices.columns])

                                    replacement = pd.DataFrame(close_prices.values, index=close_prices.index, columns=new_cols)
                                    # Use DataFrame.update to avoid shape/broadcast errors
                                    data.update(replacement)
                                else:
                                    # nothing to convert
                                    pass
                            else:
                                # ticker not present in close_prices (maybe different naming) - skip
                                pass
                        # Volume doesn't need currency conversion
                        # end fx conversion for this ticker

    if data.columns.nlevels <= 1:
        data.columns = pd.MultiIndex.from_product([data.columns, tickers])
    return data


def get_currency_returns(
    currency_list: List[str],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    base_currency: str = "USD",
    adjust: bool = True,
    raise_issue: bool = False,
) -> pd.DataFrame:
    """Retrieve currency returns versus base.

    Currencies are quoted as CCY1/CCY2 where CCY1 refers to the base currency. For ease, FX
    conventions are disregarded. For example, EUR/USD would be retrieved as USD/EUR meaning long
    USD and short EUR. Fills missing values with zeros. FX returns are returned in the same order
    as passed in.

    Args:
        currency_list (List[str]): list of currency tickers, e.g. "EUR"
        start_date (Union[str, datetime]): start date of time series
        end_date (Union[str, datetime]): end date of time series
        base_currency (str, optional): quotation basis (long). Defaults to "USD".
        adjust (bool, optional): adjust yahoo-finance request for corporate actions. Defaults to True.
        raise_issue (bool, optional): raise error if validation fails. Defaults to False.

    Returns:
        pd.DataFrame: table fo currency returns
    """
    # Convert dates to string format if needed
    start_date_str = start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d")
    end_date_str = end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d")
    
    # Create the currency pair tickers
    to_process = [f"{fx}{base_currency}=X" for fx in currency_list if fx != base_currency]
    
    # Get exchange rate data
    price_data = get_historical_price_data(
        to_process, start_date_str, end_date_str, 
        adjust=adjust, convert_to_usd=False, 
        raise_issue=raise_issue
    )
 
    # Initialize returns DataFrame
    date_range = pd.bdate_range(start_date_str, end_date_str)
    fx_returns = pd.DataFrame(index=date_range)
    zeros = pd.Series(np.zeros(date_range.shape[0]), index=date_range)
    
    # Process each currency
    for ccy in currency_list:
        if ccy == base_currency:
            fx_returns[ccy] = zeros
        else:
            ticker = f"{ccy}{base_currency}=X"
            if isinstance(price_data, pd.DataFrame) and not price_data.empty:
                # Get Close prices - handle both single and multi-level columns
                if price_data.columns.nlevels > 1:
                    close_prices = price_data["Close", ticker] if ("Close", ticker) in price_data.columns else zeros
                else:
                    close_prices = price_data["Close"] if "Close" in price_data.columns else zeros
                fx_returns[ccy] = close_prices.pct_change().fillna(0)
            else:
                fx_returns[ccy] = zeros
    
    return fx_returns


def find_ticker_currency(ticker: str, fallback: str = "USD") -> str:
    """Find currency of ticker using Yahoo! Finance website.

    If the currency cannot be found then it is assumed to be USD (or which ever fallback specified).
    Each request times out after 10 seconds.

    Args:
        ticker (str): ticker symbol

    Returns:
        str: 3 character string of currency code, e.g. ``USD``
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
            AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36",
    }

    url = f"http://finance.yahoo.com/quote/{ticker}?p={ticker}"

    try:
        return yf.Ticker(ticker).info['currency'].upper()

    except Exception as e:
        logger.warning("Error in request: %s. Returning fallback currency.", e)
        return fallback


def get_issue_currency_for_tickers(tickers: List[str]) -> List[str]:
    """Retrieve currency in which security was issued in.

    A request of 5 tickers takes around 3-4 seconds. The request is sped up using asynchronous
    execution.

    Args:
        tickers (List[str]): list of tickers

    Returns:
        List[str]: list of currency tickers, e.g. ``EUR``
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(find_ticker_currency, ticker): ticker for ticker in tickers}
        for future in concurrent.futures.as_completed(futures):
            try:
                results[futures[future]] = future.result()
            except Exception as exc:
                logger.warning(exc)
    return [results[ticker] for ticker in tickers]


def get_strategy_price_data(
    pipeline: StrategyPipeline,
    start_date: Union[str, datetime],
    end_date: Optional[Union[str, datetime]] = None,
) -> pd.DataFrame:
    """Request and store strategy data in dataframe.

    Args:
        pipeline (StrategyPipeline): strategy pipeline
        start_date (Union[str, datetime]): start date of strategies
        end_date (Union[str, datetime], optional): end date of strategies (None equals today). \
            Defaults to None.

    Returns:
        pd.DataFrame: table of adjusted prices for all strategy inputs
    """
    if end_date is None:
        end_date = datetime.today()

    all_tickers = []
    for strategy in pipeline.pipeline:
        all_tickers += strategy.get_tickers()

    start_date_str = start_date if isinstance(start_date, str) else start_date.strftime("%Y-%m-%d")
    end_date_str = end_date if isinstance(end_date, str) else end_date.strftime("%Y-%m-%d")
    data = get_historical_price_data(all_tickers, start_date_str, end_date_str, adjust=True, convert_to_usd=True)
    if isinstance(data, pd.DataFrame) and not data.empty:
        if data.columns.nlevels > 1:
            close_prices = data.xs("Close", axis=1, level=0)  # Get all Close prices
            return pd.DataFrame(close_prices)
        else:
            return pd.DataFrame(data["Close"])  # Return as DataFrame
    return pd.DataFrame()


if __name__ == "__main__":
    from src.pytaa.strategy.static import STRATEGIES

    some_pipeline = StrategyPipeline(STRATEGIES)
    print(get_strategy_price_data(some_pipeline, "2011-01-01").dropna())
