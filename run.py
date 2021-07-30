from data.data_processor import PandasDataProcessor
from data.preprocessed_data import PreprocessedData
from data.raw_data import YfinanceNSERawData
from model.model import LstmModel

ticker = 'Reliance'

model = LstmModel(ticker, PreprocessedData, PandasDataProcessor, YfinanceNSERawData)
model.train(epochs=1)
x = model.predict()
print(f'Predicted change in closing price of {ticker} is : {x}')
