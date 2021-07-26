from model.model import LstmModel

# ticker = 'Wipro'
ticker = 'Reliance'

model = LstmModel(ticker)
model.train()
# model.evaluate(dataset)
x = model.predict()
print(x)
