# Stock-Prediction
Predict stock prices using python

## Contents

- [Installation](#Installation)
- [Quickstart](#Quickstart)
- [Overview](#Overview)
  - [Model](#Model)
  - [Data](#Data)
- [Disclamer](#Disclamer)
- [Contributing](#Contributing)
 
## Installation
Clone this repo and open an instance of terminal and cd to the project's file path, e.g

```bash
cd Users/User/Desktop/Stock-Prediction
```

Then, run the following in terminal:

```bash
pip install -r requirements.txt
python run.py
```
## Quickstart

```python
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

```
(Currently, only symbols of NSE stocks can be used as a ticker)

## Overview
This project has a Model class which can be used to train a model on a stock and predict the percentage change in closing price.

### Model
`Model` -> An abstract model class.  
`LstmModel` -> Inherits from `Model` and uses keras `LSTM`.

### Data
`data` package contains classes responsible for getting raw data and processing it.

`RawData` -> Abstract class to get raw data.  
`YfinanceNSERawData` -> Inhertis from `RawData` and fetches data from [NSE](https://en.wikipedia.org/wiki/National_Stock_Exchange_of_India) using [`yfinance`](https://github.com/ranaroussi/yfinance).  

`DataProcessor` ->Abstract class for intermediate processing of raw data.  
`PandasDataProcessor` -> Inherits from `DataProcessor` and uses `pandas` and `sklearn` for intermediate processing of raw data.  

`PreprocessedData` -> For final processing of data.  

## Disclamer
This is only for educational purposes and should not be used for actual trading. I won't be resposible for your losses.

## Contributing
Feel free to fork, play around, and submit PRs. I would be very grateful 😁 for any bug fixes 🐛 or feature additions.
