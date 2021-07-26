import yfinance as yf
import pandas as pd
import os.path


NSE_COMPANY_NAME_AND_SYMBOLS_FILE_NAME = 'nse_company_name_and_symbols.csv'


def get_all_nse_company_names():
    if os.path.exists(NSE_COMPANY_NAME_AND_SYMBOLS_FILE_NAME):
        return pd.read_csv(NSE_COMPANY_NAME_AND_SYMBOLS_FILE_NAME)

    url = 'https://archives.nseindia.com/content/equities/EQUITY_L.csv'
    df = pd.read_csv(url)
    df = df[['NAME OF COMPANY', 'SYMBOL']]
    df.to_csv(NSE_COMPANY_NAME_AND_SYMBOLS_FILE_NAME, index=False)
    return df


ALLOWED_PERIOD_VALUES = {'1d', '5d', '1mo', '3mo',
                         '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'}


def get_raw_data_from_ticker(stock_ticker, period='max'):
    if period not in ALLOWED_PERIOD_VALUES:
        raise ValueError(f'period must be one of {ALLOWED_PERIOD_VALUES}')

    df = get_all_nse_company_names()

    df = df[df['SYMBOL'].str.fullmatch(stock_ticker, case=False)]
    if len(df) <= 0:
        return print('Invalid stock ticker')

    stock_ticker += '.NS'

    stock = yf.Ticker(stock_ticker)
    return stock.history(period=period)


def get_raw_data_from_name(stock_name, period='max'):
    if period not in ALLOWED_PERIOD_VALUES:
        raise ValueError(f'period must be one of {ALLOWED_PERIOD_VALUES}')

    df = get_all_nse_company_names()

    df = df[df['NAME OF COMPANY'].str.fullmatch(stock_name, case=False)]
    if len(df) <= 0:
        return print('Invalid stock name')

    stock_ticker = df['SYMBOL'].item()
    stock_ticker += '.NS'

    stock = yf.Ticker(stock_ticker)
    return stock.history(period=period)


# print(get_raw_data_from_ticker('RELIANCE', '1d'))
# print(get_raw_data_from_name('Reliance industries Limited', '1d'))

# stock = yf.Ticker("RELIANCE.NS")
# data = stock.history(period="1d")

# print(data)
