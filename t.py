from data.preprocessed_data import PreprocessedData
from model.model import LstmModel
from data.data_processor import PandasDataProcessor
from data.raw_data import YfinanceNSERawData
import matplotlib.pyplot as plt

# ticker = 'Wipro'
ticker = 'Reliance'

model = LstmModel(ticker, PreprocessedData, PandasDataProcessor, YfinanceNSERawData)
model.train(1)
# model.evaluate(dataset)
x = model.predict()
print(x)


# def show_plot(plot_data, title):
#     plt.plot(plot_data[0], label='True')
#     plt.plot(plot_data[1],  label='Predicted')
#     plt.legend()
#     plt.title(title)
#     plt.show()


# preprocessed_data = PreprocessedData(ticker, PandasDataProcessor, YfinanceNSERawData)
# scaler = preprocessed_data.data_processor.get_scaler()

# dataset_train, dataset_val, dataset_test = preprocessed_data.get_preprocessed_datasets()

# for x, y in dataset_train.take(1):
#     predictions = model.predict(x)
#     y = y.numpy()
#     for true_y, pred_y in zip(y, predictions):
#         print(true_y, pred_y)

#     show_plot([y, predictions], 'Train')
