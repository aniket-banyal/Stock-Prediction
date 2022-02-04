from data.data_processor import PandasDataProcessor
from data.keras_data.keras_preprocessed_data import KerasPreprocessedData
from data.raw_data import YfinanceNSERawData
from model.keras_models.keras_model import LstmModel

ticker = 'Reliance'

model = LstmModel(ticker, KerasPreprocessedData, PandasDataProcessor, YfinanceNSERawData, name='model_1')
model.train(epochs=1)
pred_date = '2022-02-04'
x, pred_date = model.predict(pred_date)  # To get latest prediction call with pred_date = None
print(f'Model predicts that percentage change in closing price of {ticker} on {pred_date} will be: {x}')
