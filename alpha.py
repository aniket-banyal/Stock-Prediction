from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='ER0NX6OQN1DXPANL', output_format='pandas')
data, meta_data = ts.get_intraday('BSE:RELIANCE')
print(data, meta_data)
