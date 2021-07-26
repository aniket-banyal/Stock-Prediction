from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
from .get_raw_data import get_raw_df_from_ticker
from .constants import STOCK_CSV_BASE_PATH, FUTURE_PERIOD_PREDICT, FEATURE_KEYS
from tensorflow import keras
from model.constants import SEQ_LEN, STEP, BATCH_SIZE


def get_preprocessed_datasets(ticker: str):
    (train_x, train_y), (val_x, val_y),  (test_x, test_y) = get_preprocessed_data(ticker)

    dataset_train = get_dataset(train_x, train_y)
    dataset_val = get_dataset(val_x, val_y)
    dataset_test = get_dataset(test_x, test_y)

    return dataset_train, dataset_val, dataset_test


def get_preprocessed_data(ticker: str):
    df = get_stock_df_from_ticker(ticker)
    df = df[FEATURE_KEYS]

    train_data, test_data, val_data = get_split_train_val_test_df(df)

    train_x, train_y, scaler = preprocess_df(train_data)
    val_x, val_y, _ = preprocess_df(val_data, scaler)
    test_x, test_y, _ = preprocess_df(test_data, scaler)

    return (train_x, train_y), (val_x, val_y),  (test_x, test_y)


def get_split_train_val_test_df(df: pd.DataFrame):
    train_split_fraction = 0.8
    train_split = int(train_split_fraction * len(df))

    train_data = df[: train_split]
    test_data = df[train_split:]

    val_split_fraction = 0.9
    val_split = int(val_split_fraction * len(train_data))

    val_data = train_data[val_split:]
    train_data = train_data[:val_split]
    return train_data, test_data, val_data


def preprocess_df(df: pd.DataFrame, scaler=None, return_y: bool = True):
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

    return df, scaler


def get_dataset(x: pd.DataFrame, y: pd.DataFrame, seq_len: int = SEQ_LEN, step: int = STEP, batch_size: int = BATCH_SIZE):
    dataset = keras.preprocessing.timeseries_dataset_from_array(
        x, y,
        sequence_length=seq_len,
        sampling_rate=step,
        batch_size=batch_size,
    )
    return dataset


def get_stock_df_from_ticker(ticker: str) -> pd.DataFrame:
    dirname = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(dirname, STOCK_CSV_BASE_PATH)
    file_path = os.path.join(base_path, ticker + '.csv')

    if not os.path.exists(base_path):
        os.mkdir(base_path)

    if not os.path.exists(file_path):
        df = get_raw_df_from_ticker(ticker)
        df.to_csv(file_path)

    df = pd.read_csv(file_path)
    return df

# return whole df for evaluation


def get_preprocessed_evaluation_df(stock, days=SEQ_LEN):
    df = get_stock_df_from_ticker(stock)
    df = df[FEATURE_KEYS]

    # During preprocessing some rows will be dropped, bt we want exactly SEQ_LEN number of rows and hence use x.tail() after preprocessing
    x, y, _ = preprocess_df(df)
    x = x.tail(days)
    dataset = get_dataset(x, y, batch_size=1)
    return dataset


def get_preprocessed_prediction_df(stock, days=SEQ_LEN):
    df = get_stock_df_from_ticker(stock)
    df = df[FEATURE_KEYS]

    # During preprocessing some rows will be dropped, bt we want exactly SEQ_LEN number of rows and hence use x.tail() after preprocessing
    x, scaler = preprocess_df(df, return_y=False)
    x = x.tail(days)
    x = np.expand_dims(x, 0)
    return x, scaler
