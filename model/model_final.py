from .constants import SAVED_MODELS_BASE_PATH
from .create_model import create_model
import os
import matplotlib.pyplot as plt
from data.preprocess_data import get_preprocessed_datasets
from tensorflow.python.keras.callbacks import ModelCheckpoint


def train_model(ticker):
    epochs = 1

    dataset_train, dataset_val, dataset_test = get_preprocessed_datasets(ticker)

    for batch in dataset_train.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)

    input_shape = (inputs.shape[1], inputs.shape[2])
    model = create_model(input_shape)
    checkpoint_base_path = os.path.join(SAVED_MODELS_BASE_PATH, ticker)
    # checkpoint_path = os.path.join(checkpoint_base_path, '{epoch:02d}-{val_loss:.4f}.ckpt')
    checkpoint_path = os.path.join(checkpoint_base_path, 'cp.ckpt')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                 save_best_only=True, save_weights_only=True)

    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[checkpoint]
    )

    print('History: ', history.history)

    print('Test Score: ', model.evaluate(dataset_test))
    # def visualize_loss(history, title):
    #     loss = history.history["loss"]
    #     val_loss = history.history["val_loss"]
    #     epochs = range(len(loss))
    #     plt.figure()
    #     plt.plot(epochs, loss, "b", label="Training loss")
    #     plt.plot(epochs, val_loss, "r", label="Validation loss")
    #     plt.title(title)
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Loss")
    #     plt.legend()
    #     plt.show()

    # visualize_loss(history, "Training and Validation Loss")

    # def show_plot(plot_data):
    #     plt.plot(plot_data[0], label='True')
    #     plt.plot(plot_data[1],  label='Predicted')
    #     plt.legend()
    #     plt.show()

    # for x, y in dataset_train.take(2):
    #     predictions = model.predict(x)
    #     show_plot([y.numpy(), predictions])

    # for x, y in dataset_val.take(2):
    #     predictions = model.predict(x)
    #     show_plot([y.numpy(), predictions])

    # for x, y in dataset_test.take(2):
    #     predictions = model.predict(x)
    #     show_plot([y.numpy(), predictions])


# ticker = 'reliance'
# train_model(ticker)
