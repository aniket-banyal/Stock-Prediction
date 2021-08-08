from abc import ABC, abstractmethod
from typing import Type

import utils.utils as ut
from data.data_processor import DataProcessor
from data.preprocessed_data import PreprocessedData
from data.raw_data import RawDataSource


class ModelNotFoundError(Exception):
    def __init__(self, ticker: str, seq_len: int, step: int) -> None:
        msg = f"There is no saved model for '{ticker}' with seq_len={seq_len} and step={step}. Please train such a model first."
        super().__init__(msg)


class Model(ABC):
    def __init__(self, ticker: str, preprocessed_data: Type[PreprocessedData],
                 data_processor: Type[DataProcessor], raw_data_source: Type[RawDataSource]) -> None:
        ut.validate_ticker(ticker)
        self.ticker = ticker
        self.preprocessed_data = preprocessed_data(ticker, data_processor, raw_data_source)

    @abstractmethod
    def train(self, epochs: int = 1):
        """Method to train a model"""

    @abstractmethod
    def predict(self, date: str):
        """Method to give prediction for a date"""
