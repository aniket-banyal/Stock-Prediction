import os

import pandas as pd

from data.constants import (NAME_OF_COMP_COLUMN,
                            NSE_COMPANY_NAME_AND_SYMBOLS_FILE_NAME,
                            SYMBOL_COLUMN)


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
