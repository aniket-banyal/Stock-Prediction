import os
from abc import ABC, abstractmethod
import datetime as dt
from typing import List, Type

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data.constants import FUTURE_PERIOD_PREDICT
from data.raw_data import RawDataSource


# have another dataprocessor class if you want data of multiple tickers to be used for training
class DataProcessor(ABC):
    def __init__(self, ticker: str, raw_data_source: Type[RawDataSource]) -> None:
        self.raw_data_source = raw_data_source(ticker)

    @ abstractmethod
    def get_preprocessed_dfs(self):
        """Returns train, val and test dataframes"""

    @ abstractmethod
    def get_preprocessed_prediction_df(self, date: dt.date):
        """Returns a preprocessed dataframe to be used for prediction"""


class PandasDataProcessor(DataProcessor):
    TEST_SPLIT_FRACTION = 0.2
    VAL_SPLIT_FRACTION = 0.2
    SCALER_FILE_NAME = 'scaler.gz'

    def get_preprocessed_dfs(self) -> List[pd.DataFrame]:
        df = self.raw_data_source.get_raw_df()
        df = df[self.raw_data_source.FEATURE_KEYS]

        train_data, val_data, test_data = self.train_val_test_split(df)

        train_x, train_y, scaler = self.get_preprocessed_df(train_data)
        val_x, val_y, _ = self.get_preprocessed_df(val_data, scaler)
        test_x, test_y, _ = self.get_preprocessed_df(test_data, scaler)

        self.save_scaler(scaler)

        return (train_x, train_y), (val_x, val_y),  (test_x, test_y)

    def get_preprocessed_df(self, df: pd.DataFrame, scaler=None, return_y: bool = True):
        df = df.astype(np.float64)
        df.dropna(inplace=True)

        df = df[df[self.raw_data_source.VOL_COLUMN] != 0]

        df = df.pct_change()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(inplace=True, drop=True)

        if scaler is None:
            scaler = StandardScaler().fit(df)
        df = pd.DataFrame(scaler.transform(df), columns=self.raw_data_source.FEATURE_KEYS)

        if return_y:
            df['target'] = df[self.raw_data_source.CLOSE_COLUMN].shift(-FUTURE_PERIOD_PREDICT)
            df.dropna(inplace=True)
            return df[self.raw_data_source.FEATURE_KEYS], df['target'], scaler

        return df, scaler

    def get_preprocessed_prediction_df(self, date: dt.date, seq_len) -> pd.DataFrame:
        # after preprocessing data, some rows will get dropped, so that's why x=x.tail(seq_len) needs to be done after preprocessing
        df = self.raw_data_source.get_raw_df()

        df_last_date = df.index[-1].date()
        pred_date = self.get_correct_pred_date(date, df_last_date)

        # in loc the end index is included so subtracting 1
        end_date = pred_date - dt.timedelta(days=1)

        df = df.loc[:end_date]

        df = df[self.raw_data_source.FEATURE_KEYS]
        scaler = self.get_scaler()
        x, _ = self.get_preprocessed_df(df, scaler=scaler, return_y=False)
        x = x.tail(seq_len)
        return x, pred_date

    def get_correct_pred_date(self, date: dt.date, df_last_date: dt.date):
        """
        If df_last_date is Fri and date is Sat or Sun then the actual pred_date is of next Mon

        If df_last_date >= date - dt.timedelta(days=1) then we surely have the data for that pred_date
        """
        if df_last_date.weekday() == 4:
            # If df_last_date if Fri and date is of Sat
            if date == df_last_date + dt.timedelta(days=1):
                date += dt.timedelta(days=2)
            # If df_last_date if Fri and date is of Sun
            elif date == df_last_date + dt.timedelta(days=2):
                date += dt.timedelta(days=1)
            # If df_last_date is Fri and date is greater than next Mon
            elif date > df_last_date + dt.timedelta(days=3):
                raise InvalidPredictionDateError(date, df_last_date)

        elif df_last_date < date - dt.timedelta(days=1):
            raise InvalidPredictionDateError(date, df_last_date)

        return date

    def train_val_test_split(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        test_split = int((1-self.TEST_SPLIT_FRACTION) * len(df))
        x = df[: test_split]
        test_data = df[test_split:]

        val_split = int((1-self.VAL_SPLIT_FRACTION) * len(x))
        train_data = x[:val_split]
        val_data = x[val_split:]
        return train_data, val_data, test_data

    def save_scaler(self, scaler: StandardScaler) -> None:
        file_path = self.get_scaler_file_path()
        joblib.dump(scaler, file_path)

    def get_scaler(self) -> StandardScaler:
        file_path = self.get_scaler_file_path()
        return joblib.load(file_path)

    def get_scaler_file_path(self):
        base_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(base_path, self.SCALER_FILE_NAME)
        return file_path


class InvalidPredictionDateError(Exception):
    def __init__(self, pred_date: dt.date, df_last_date: dt.date) -> None:
        msg = f"We have data till {df_last_date}, so model can predict only till {df_last_date + dt.timedelta(days=1)}. But you called model's predict function with this prediction date: {pred_date}"
        super().__init__(msg)
