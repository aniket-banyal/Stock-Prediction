from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential


# def create_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())

#     model.add(LSTM(128, return_sequences=True))
#     model.add(Dropout(0.1))
#     model.add(BatchNormalization())

#     model.add(LSTM(128, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())

#     model.add(LSTM(128))
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())

#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.2))

#     model.add(Dense(1))

#     model.compile(
#         loss='mse',
#         optimizer='adam',
#     )

#     return model


def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    model.compile(
        loss='mse',
        optimizer='adam',
    )

    return model
