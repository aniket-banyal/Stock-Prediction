from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
from .get_raw_data import get_raw_data_from_ticker
from .constants import STOCK_CSV_BASE_PATH, FUTURE_PERIOD_PREDICT, FEATURE_KEYS
from tensorflow import keras
from model.constants import SEQ_LEN, STEP, BATCH_SIZE


def get_preprocessed_datasets(stock):
    (train_x, train_y), (val_x, val_y),  (test_x, test_y) = get_preprocessed_data(stock)

    dataset_train = get_dataset(train_x, train_y)
    dataset_val = get_dataset(val_x, val_y)
    dataset_test = get_dataset(test_x, test_y)

    return dataset_train, dataset_val, dataset_test


def get_preprocessed_data(stock):
    df = get_stock_df(stock)
    df = df[FEATURE_KEYS]

    train_split_fraction = 0.8
    train_split = int(train_split_fraction * len(df))

    train_data = df[: train_split]
    test_data = df[train_split:]

    val_split_fraction = 0.9
    val_split = int(val_split_fraction * len(train_data))

    val_data = train_data[val_split:]
    train_data = train_data[:val_split]

    train_x, train_y, scaler = preprocess_df(train_data)
    val_x, val_y, _ = preprocess_df(val_data, scaler)
    test_x, test_y, _ = preprocess_df(test_data, scaler)

    return (train_x, train_y), (val_x, val_y),  (test_x, test_y)


def get_preprocessed_evaluation_df(stock, days=SEQ_LEN):
    df = get_stock_df(stock)
    df = df[FEATURE_KEYS]

    # During preprocessing some rows will be dropped, bt we want exactly SEQ_LEN number of rows and hence use x.tail() after preprocessing
    x, y, _ = preprocess_df(df)
    x = x.tail(days)
    dataset = get_dataset(x, y, batch_size=1)
    return dataset


def get_preprocessed_prediction_df(stock, days=SEQ_LEN):
    df = get_stock_df(stock)
    df = df[FEATURE_KEYS]
    x, scaler = preprocess_df(df, return_y=False)
    x = x.tail(days)
    # x = x[-SEQ_LEN-1:len(x)-1]
    # print(x)
    # print(scaler.inverse_transform(x))
    x = np.expand_dims(x, 0)
    return x, scaler


def get_stock_df(stock):
    file_name = os.path.join(STOCK_CSV_BASE_PATH, stock + '.csv')

    if not os.path.exists(STOCK_CSV_BASE_PATH):
        os.mkdir(STOCK_CSV_BASE_PATH)

    if not os.path.exists(file_name):
        df = get_raw_data_from_ticker(stock)
        df.to_csv(file_name)

    df = pd.read_csv(file_name)
    # print(df.tail(10))
    return df


def preprocess_df(df, scaler=None, return_y=True):
    df = df.astype(np.float64)
    df.dropna(inplace=True)

    df = df[df['Volume'] != 0]
    df = df.pct_change()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)

    if scaler is None:
        scaler = StandardScaler().fit(df)
    df = pd.DataFrame(scaler.transform(df), columns=FEATURE_KEYS)

    if return_y:
        df['target'] = df['Close'].shift(-FUTURE_PERIOD_PREDICT)
        df.dropna(inplace=True)

        return df[FEATURE_KEYS], df['target'], scaler

    # print(df.tail(10))

    return df, scaler


def get_dataset(x, y, seq_len=SEQ_LEN, step=STEP, batch_size=BATCH_SIZE):
    dataset = keras.preprocessing.timeseries_dataset_from_array(
        x, y,
        sequence_length=seq_len,
        sampling_rate=step,
        batch_size=batch_size,
    )
    return dataset
