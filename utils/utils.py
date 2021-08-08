import datetime as dt
import os

import pandas as pd

NSE_COMPANY_NAME_AND_SYMBOLS_FILE_NAME = 'nse_company_name_and_symbols.csv'
NAME_OF_COMP_COLUMN = 'NAME OF COMPANY'
SYMBOL_COLUMN = 'SYMBOL'
DATE_FORMAT = '%Y-%m-%d'


class InvalidTickerError(Exception):
    def __init__(self, ticker: str) -> None:
        msg = f"'{ticker}' is not a valid stock ticker"
        super().__init__(msg)


class InvalidPredictionDateError(Exception):
    def __init__(self, pred_date: dt.date, df_date: dt.date, seq_len: int, lower: bool = False) -> None:
        if lower:
            msg = f"We have data till {df_date} and model needs sequences of length {seq_len} to predict. So model can predict only for dates starting from {df_date + dt.timedelta(days=seq_len)}. But you called model's predict function with this prediction date: {pred_date}"
        else:
            msg = f"We have data till {df_date}, so model can predict only till {df_date + dt.timedelta(days=1)}. But you called model's predict function with this prediction date: {pred_date}"
        super().__init__(msg)


class InvalidDateError(Exception):
    def __init__(self, date: dt.date) -> None:
        msg = f"{date} is not in correct format. Please give date in {DATE_FORMAT} format."
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


def get_date_from_string(date: str) -> dt.date:
    try:
        return dt.datetime.strptime(date, DATE_FORMAT).date()
    except ValueError:
        raise InvalidDateError(date)


def get_prediction_date(df: pd.DataFrame, seq_len: int, date: str = None):
    if date is None:
        pred_date = dt.datetime.now().date()
        # timezone = pytz.timezone("Asia/Kolkata")
    else:
        pred_date = get_date_from_string(date)

    df_first_date = df.index[0].date()
    df_last_date = df.index[-1].date()
    return get_correct_pred_date(pred_date, df_first_date, df_last_date, seq_len)


def get_correct_pred_date(pred_date: dt.date, df_first_date: dt.date, df_last_date: dt.date,  seq_len: int):
    """
    If df_last_date is Fri and date is Sat or Sun then the actual pred_date is of next Mon

    If df_last_date >= date - dt.timedelta(days=1) then we surely have the data for that pred_date
    """

    if df_first_date + dt.timedelta(days=seq_len) > pred_date:
        raise InvalidPredictionDateError(pred_date, df_first_date, lower=True)

    # If df_last_date is Fri and pred_date is greater than next Mon
    if df_last_date.weekday() == 4:
        if pred_date > df_last_date + dt.timedelta(days=3):
            raise InvalidPredictionDateError(pred_date, df_last_date)

    elif df_last_date < pred_date - dt.timedelta(days=1):
        raise InvalidPredictionDateError(pred_date, df_last_date)

    # If date is Sat or Sun make it Mon
    if pred_date.weekday() == 5:
        pred_date += dt.timedelta(days=2)

    elif pred_date.weekday() == 6:
        pred_date += dt.timedelta(days=1)
    return pred_date
