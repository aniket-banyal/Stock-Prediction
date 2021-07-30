from collections import deque
from typing import Type

import numpy as np
import pandas as pd
from model.constants import BATCH_SIZE, SEQ_LEN, STEP
from tensorflow import keras

from data.data_processor import DataProcessor
from data.raw_data import RawDataSource


class PreprocessedData:
    def __init__(self, ticker: str, data_processor: Type[DataProcessor], raw_data_source: Type[RawDataSource]) -> None:
        # validate_ticker(ticker)
        self.data_processor = data_processor(ticker, raw_data_source)

    def get_preprocessed_datasets(self):
        (train_x, train_y), (val_x, val_y),  (test_x, test_y) = self.data_processor.get_preprocessed_dfs()

        dataset_train = self.get_dataset_from_df(train_x, train_y)
        dataset_val = self.get_dataset_from_df(val_x, val_y)
        dataset_test = self.get_dataset_from_df(test_x, test_y)

        return dataset_train, dataset_val, dataset_test

    def get_dataset_from_df(self, x: pd.DataFrame, y: pd.DataFrame, seq_len: int = SEQ_LEN,
                            step: int = STEP, batch_size: int = BATCH_SIZE):

        t = []
        for i in range(SEQ_LEN-1, len(y), SEQ_LEN):
            t.append(y[i])

        dataset = keras.preprocessing.timeseries_dataset_from_array(
            x, t,
            sequence_length=seq_len,
            sampling_rate=step,
            batch_size=batch_size,
        )
        return dataset

        #     print(i, ((SEQ_LEN-1) + SEQ_LEN*3))
        #     if (i % ((SEQ_LEN-1) + SEQ_LEN*3) == 0):
        #         print(i)
        #         print(x[i: i+10])
        #     print(y[i])
        #     print()

        # print()
        # print('Y')
        # print(y[SEQ_LEN-1: SEQ_LEN*5])
        # print()

        # sequential_data = []
        # prev_days = deque(maxlen=SEQ_LEN)
        # # print(x[SEQ_LEN-1:])
        # print()
        # print('T: ')
        # print(t)
        # print()

        # j = 0
        # for i in x.values:
        #     prev_days.append([n for n in i])
        #     if len(prev_days) == SEQ_LEN and j < len(t):
        #         sequential_data.append([np.array(prev_days), t[j]])
        #         j += 1

        # X = []
        # y = []

        # for seq, target in sequential_data:  # going over our new sequential data
        #     X.append(seq)  # X is the sequences
        #     y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

        # print()
        # print('SENTDEX')
        # print()
        # for i in range(3):
        #     print(X[i][0], y[i])

        # print()
        # print()
        # print()

        # print()
        # print('KERAS')
        # print()
        # # return np.array(X), y
        # # random.shuffle(sequential_data)

        # print('##############################################')
        # print(len(x), len(y), len(y)//SEQ_LEN, len(x)/SEQ_LEN//batch_size)
        # i = 0
        # for x, y in dataset:
        #     i += 1
        # print(i)
        # print('y', len(y))
        # # print('y', y)
        # # print('x', x)
        # #     for i in range(1):
        # #         print('x', x[i][0])
        # print('##############################################')

    def get_preprocessed_prediction_dataset(self, seq_len: int = SEQ_LEN,
                                            step: int = STEP, batch_size: int = BATCH_SIZE):

        x = self.data_processor.get_preprocessed_prediction_df(seq_len)
        dataset = keras.preprocessing.timeseries_dataset_from_array(
            x,
            targets=None,
            sequence_length=seq_len,
            sampling_rate=step,
            batch_size=batch_size,
        )
        return dataset

        # x = np.expand_dims(x, 0)
        # return x, scaler
