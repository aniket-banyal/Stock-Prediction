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
python t.py
```
## Quickstart

```python
from data.preprocessed_data import PreprocessedData
from model.model import LstmModel
from data.data_processor import PandasDataProcessor
from data.raw_data import YfinanceNSERawData

ticker = 'Reliance'
model = LstmModel(ticker, PreprocessedData, PandasDataProcessor, YfinanceNSERawData)
model.train(epochs=10)
x = model.predict()
print(f'Predicted change in closing price of {ticker} is - {x}')
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
