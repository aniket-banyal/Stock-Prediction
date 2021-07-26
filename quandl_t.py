import quandl
quandl.ApiConfig.api_key = "oCx5MwsuoLtXwEXNe2wX"
# data = quandl.get('NSE/RELIANCE', start_date="2001-01-01",
#                   end_date="2021-07-07")
data = quandl.get('NSE/RELIANCE', rows=5, sort_order="desc")
# data = quandl.get_table('ZACKS/FC', ticker='AAPL')
print(data)
