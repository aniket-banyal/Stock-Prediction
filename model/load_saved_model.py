from data.constants import FEATURE_KEYS
from data.preprocess_data import get_preprocessed_evaluation_df, get_preprocessed_prediction_df
from .constants import SAVED_MODELS_BASE_PATH, SEQ_LEN
from .create_model import create_model
import os
import pandas as pd
import numpy as np


def get_model(ticker):
    ticker = 'Reliance'

    # input_shape = (inputs.shape[1], inputs.shape[2])
    input_shape = (SEQ_LEN, len(FEATURE_KEYS))

    checkpoint_base_path = os.path.join(SAVED_MODELS_BASE_PATH, ticker)
    checkpoint_path = os.path.join(checkpoint_base_path, 'cp.ckpt')

    model = create_model(input_shape)
    # print(model.evaluate(dataset_test))
    model.load_weights(checkpoint_path).expect_partial()
    # print(model.evaluate(dataset_test))

    return model


def evaluate(ticker):
    model = get_model(ticker)
    dataset = get_preprocessed_evaluation_df(ticker)
    print(model.evaluate(dataset))


def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values


def predict(ticker):
    model = get_model(ticker)
    x, scaler = get_preprocessed_prediction_df(ticker)
    y = model.predict(x)[0]
    actual_y = invTransform(scaler, y, 'Close', FEATURE_KEYS)[0]

    return actual_y*100
