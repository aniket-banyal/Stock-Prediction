import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Type

import numpy as np
import pandas as pd
import pytz
from data.data_processor import DataProcessor
from data.preprocessed_data import PreprocessedData
from data.raw_data import RawDataSource
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import errors_impl
from tensorflow.python.keras.callbacks import ModelCheckpoint
from utils.utils import validate_ticker

from .constants import SAVED_MODELS_BASE_PATH, SEQ_LEN


class ModelNotFoundError(Exception):
    def __init__(self, ticker: str) -> None:
        msg = f"There is no saved model for '{ticker}'. Please train a model for '{ticker}' first."
        super().__init__(msg)


class Model(ABC):
    def __init__(self, ticker: str, preprocessed_data: Type[PreprocessedData],
                 data_processor: Type[DataProcessor], raw_data_source: Type[RawDataSource]) -> None:
        validate_ticker(ticker)
        self.ticker = ticker
        self.preprocessed_data = preprocessed_data(ticker, data_processor, raw_data_source)
        self.input_shape = (SEQ_LEN, len(self.preprocessed_data.data_processor.raw_data_source.FEATURE_KEYS))

    def train(self, epochs: int = 1):
        dataset_train, dataset_val, dataset_test = self.preprocessed_data.get_preprocessed_datasets()

        for batch in dataset_train.take(1):
            inputs, targets = batch
        print("Input shape:", inputs.numpy().shape)
        print("Target shape:", targets.numpy().shape)

        input_shape = (inputs.shape[1], inputs.shape[2])
        assert input_shape == self.input_shape

        model = self.__get_model()

        checkpoint_path = get_checkpoint_path(self.ticker)
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                     save_best_only=True, save_weights_only=True)

        history = model.fit(
            dataset_train,
            epochs=epochs,
            validation_data=dataset_val,
            callbacks=[checkpoint]
        )

        print('History: ', history.history)
        print('Test Loss: ', model.evaluate(dataset_test))

    def __get_model(self):
        try:
            model = self.__load_saved_model()
        except ModelNotFoundError:
            model = self._create_model()
        return model

    def __load_saved_model(self):
        # the weights of the loaded model can be different from the weights of the model in train cuz only the best weights are saved.
        checkpoint_path = get_checkpoint_path(self.ticker)
        model = self._create_model()
        # latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_path))
        try:
            model.load_weights(checkpoint_path)
        except errors_impl.NotFoundError:
            raise ModelNotFoundError(self.ticker)
        return model

    def predict(self, date: str = None):
        pred_date = self.__get_prediction_date(date)

        model = self.__load_saved_model()
        x = self.preprocessed_data.get_preprocessed_prediction_dataset(pred_date)
        scaler = self.preprocessed_data.data_processor.get_scaler()
        y = model.predict(x)
        actual_y = self.__invTransform(scaler, y, self.preprocessed_data.data_processor.raw_data_source.CLOSE_COLUMN, self.preprocessed_data.data_processor.raw_data_source.FEATURE_KEYS)[0]

        return actual_y*100, pred_date

    def __get_prediction_date(self, date: str = None):
        if date is None:
            pred_date = datetime.now().date()
            # timezone = pytz.timezone("Asia/Kolkata")
        else:
            pred_date = datetime.strptime(date, '%Y-%m-%d').date()
        return pred_date

    @staticmethod
    def __invTransform(scaler, data, colName, colNames):
        dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
        dummy[colName] = data
        dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
        return dummy[colName].values

    @abstractmethod
    def _create_model():
        pass


# you could have various LstmModels by having their own STEP, SEQ_LEN
class LstmModel(Model):

    def _create_model(self):
        model = Sequential()
        model.add(LSTM(256, input_shape=self.input_shape, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(1))

        model.compile(
            loss='mse',
            optimizer='adam',
        )

        return model


def get_checkpoint_path(ticker):
    dirname = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(dirname, SAVED_MODELS_BASE_PATH)
    checkpoint_base_path = os.path.join(base_path, ticker)
    checkpoint_path = os.path.join(checkpoint_base_path, 'cp.ckpt')

    return checkpoint_path
