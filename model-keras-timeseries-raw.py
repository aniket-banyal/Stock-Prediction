from constants import FEATURE_KEYS
import os
from preprocess_data import get_stock_df
import pandas as pd
from get_raw_data import get_raw_data_from_ticker
import matplotlib.pyplot as plt
from tensorflow import keras

ticker = 'reliance'
df = get_stock_df(ticker)

# print(df)

features = df[FEATURE_KEYS]

split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
step = 1
past = 60
future = 1
batch_size = 64
epochs = 3


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


features = normalize(features.values, train_split)
features = pd.DataFrame(features, columns=FEATURE_KEYS)

train_data = features.loc[0: train_split - 1]
val_data = features.loc[train_split:]

start = past + future
end = start + train_split

x_train = train_data.values
print(x_train)
y_train = features.iloc[start:end]['Close']
print('y_train', y_train)

sequence_length = int(past / step)


dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.iloc[:x_end].values
y_val = features.iloc[label_start:]['Close']

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


for batch in dataset_train.take(1):
    inputs, targets = batch

print("Input shape:", inputs.numpy().shape)
print("Target shape:", targets.numpy().shape)

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss="mse")
# model.summary()

history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
)

print(history.history)


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# visualize_loss(history, "Training and Validation Loss")


def show_plot(plot_data):
    plt.plot(plot_data[0])
    plt.plot(plot_data[1])
    # plt.legend()
    # plt.xlim([time_steps[0], (future + 5) * 2])
    # plt.xlabel("Time-Step")
    plt.show()


for x, y in dataset_train.take(2):
    predictions = model.predict(x)
    show_plot([y.numpy(), predictions])


for x, y in dataset_val.take(2):
    predictions = model.predict(x)
    # for i in range(len(y)):
    #     print(y[i].numpy(), predictions[i])

    show_plot([y.numpy(), predictions])
