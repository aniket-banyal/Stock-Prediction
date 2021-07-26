import pandas as pd
import yfinance as yf
from utils.utils import validate_ticker

from .constants import FEATURE_KEYS

ALLOWED_PERIOD_VALUES = {'1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'}


class FeatureColNotPresentInDfError(Exception):
    def __init__(self, column: str) -> None:
        msg = f"'{column}' column not present in df received from yfinance"
        super().__init__(msg)


def get_raw_df_from_ticker(stock_ticker: str, period: str = 'max') -> pd.DataFrame:
    if period not in ALLOWED_PERIOD_VALUES:
        raise ValueError(f'period must be one of {ALLOWED_PERIOD_VALUES}')

    validate_ticker(stock_ticker)

    stock_ticker += '.NS'

    stock = yf.Ticker(stock_ticker)
    df = stock.history(period=period)

    for column in FEATURE_KEYS:
        if column not in df.columns:
            raise FeatureColNotPresentInDfError(column)

    return df

# def get_raw_df_from_name(stock_name: str, period: str = 'max') -> pd.DataFrame:
#     if period not in ALLOWED_PERIOD_VALUES:
#         raise ValueError(f'period must be one of {ALLOWED_PERIOD_VALUES}')

#     df = get_all_nse_company_names_and_ticker()

#     df = df[df[NAME_OF_COMP_COLUMN].str.fullmatch(stock_name, case=False)]
#     if len(df) <= 0:
#         raise StockNameNotFoundError(stock_name)

#     stock_ticker = df['SYMBOL'].item()
#     stock_ticker += '.NS'

#     stock = yf.Ticker(stock_ticker)
#     return stock.history(period=period)


# print(get_raw_df_from_ticker('RELIANCE'))
# print(get_raw_df_from_name('Reliance industries Limited', '1d'))
