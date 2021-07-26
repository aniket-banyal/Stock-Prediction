# from model.model_final import train_model
from model.load_saved_model import evaluate, predict

ticker = 'reliance'
# evaluate(ticker)
pred = predict(ticker)
print(pred)
# -0.18728648994287966

# train_model(ticker)
