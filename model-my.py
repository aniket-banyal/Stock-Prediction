import numpy as np
import pandas as pd
from constants import FEATURE_KEYS, SEQ_LEN
from preprocess_data import get_preprocessed_data

STOCK_TO_PREDICT = "Reliance"

EPOCHS = 3
BATCH_SIZE = 64

# NAME = f"{STOCK_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
# tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

(train_x, train_y), (val_x, val_y),  (test_x, test_y) = get_preprocessed_data(STOCK_TO_PREDICT)
print(train_x.shape, train_y.shape, val_x.shape, val_y.shape, test_x.shape, test_y.shape)
# print(train_x[0], train_y[0])
# print(val_x[0], val_y[0])

INPUT_SHAPE = train_x.shape[1:]

if __name__ == '__main__':
    import tensorflow as tf
    from tensorflow import keras
    import matplotlib.pyplot as plt
    import os.path
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.callbacks import TensorBoard
    import os
    from create_model import create_model

    checkpoint_base_path = os.path.join('model', STOCK_TO_PREDICT)
    checkpoint_path = os.path.join(checkpoint_base_path, '{epoch:02d}-{val_loss:.4f}.ckpt')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                 save_best_only=True, save_weights_only=True)

    model = create_model(INPUT_SHAPE)

    history = model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(val_x, val_y),
        callbacks=[checkpoint],
    )

    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score)

    test_x_last = test_x[-1]
    test_x_last = pd.DataFrame(test_x_last, columns=FEATURE_KEYS)
    # print(test_x_last)
    test_x_last['Close'].plot()

    test_x_last = np.expand_dims(test_x_last, 0)
    pred = model.predict(test_x_last).flatten()
    for i in range(len(test_x_last[0])):
        print(pred[i], test_x_last[0][i][3])

    # print(pred)
    plt.plot(pred)

    plt.show()

    # model = create_model(INPUT_SHAPE)

    # score = model.evaluate(val_x, val_y, verbose=0)
    # print('Untrained Test loss:', score)

    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # print(latest)

    # model.load_weights(latest)

    # score = model.evaluate(val_x, val_y, verbose=0)
    # print('Load Test loss:', score)
