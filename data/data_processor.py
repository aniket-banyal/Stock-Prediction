import os
from abc import ABC, abstractmethod
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
    def get_preprocessed_prediction_df(self):
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

    def get_preprocessed_prediction_df(self, seq_len) -> pd.DataFrame:
        df = self.raw_data_source.get_raw_df()
        df = df[self.raw_data_source.FEATURE_KEYS]
        scaler = self.get_scaler()
        x, _ = self.get_preprocessed_df(df, scaler=scaler, return_y=False)
        x = x.tail(seq_len)
        return x

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
