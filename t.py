import matplotlib.pyplot as plt

from data.data_processor import PandasDataProcessor
from data.preprocessed_data import PreprocessedData
from data.raw_data import YfinanceNSERawData
from model.model import LstmModel

# ticker = 'Wipro'
ticker = 'Reliance'

model = LstmModel(ticker, PreprocessedData, PandasDataProcessor, YfinanceNSERawData)
model.train(1)
# model.evaluate(dataset)
# date = '2021-08-06'
# date = '2021-08-07'
# date = '2021-08-08'
date = '2021-08-09'
# date = '1996-03-31'
# date = '1996-03-30'
# date = '1996-03-31'
x, pred_date = model.predict(date)
print(f'Predicted change in closing price of {ticker} for date: {pred_date} is : {x}')


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
