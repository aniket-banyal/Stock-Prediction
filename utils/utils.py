import os

import pandas as pd

NSE_COMPANY_NAME_AND_SYMBOLS_FILE_NAME = 'nse_company_name_and_symbols.csv'
NAME_OF_COMP_COLUMN = 'NAME OF COMPANY'
SYMBOL_COLUMN = 'SYMBOL'


class InvalidTickerError(Exception):
    def __init__(self, ticker: str) -> None:
        msg = f"'{ticker}' is not a valid stock ticker"
        super().__init__(msg)


def get_all_nse_company_names_and_ticker() -> pd.DataFrame:
    base_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(base_path, NSE_COMPANY_NAME_AND_SYMBOLS_FILE_NAME)

    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    url = 'https://archives.nseindia.com/content/equities/EQUITY_L.csv'
    df = pd.read_csv(url)
    df = df[[NAME_OF_COMP_COLUMN, SYMBOL_COLUMN]]
    df.to_csv(file_path, index=False)
    return df


def validate_ticker(ticker):
    df = get_all_nse_company_names_and_ticker()

    df = df[df[SYMBOL_COLUMN].str.fullmatch(ticker, case=False)]
    if len(df) <= 0:
        raise InvalidTickerError(ticker)
