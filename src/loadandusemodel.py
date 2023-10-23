import pprint
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
import helper_bm as helper
import time
import os
from sklearn.metrics import mean_squared_error
import numpy as np

#model = load_model('./src/model.h5')
model = load_model('Tradingtools/Stock-Price-Prediction/src/model.h5')
model = load_model('./src/model.h5')
#model = load_model('Tradingtools/Stock-Price-Prediction/src/model.h5')

seq_len = 50

#X_te, orig_data = helper.load_data('../last50_sp500_prices.csv', seq_len, True)
X_te, orig_data = helper.load_data('Tradingtools/last50_sp500_prices.csv', seq_len, True)
X_te = np.reshape(X_te, (1,seq_len, 1))
X_te = helper.load_data('../last50_sp500_prices.csv', seq_len, True)
#X_te = helper.load_data('Tradingtools/last50_sp500_prices.csv', seq_len, True)

win_size = seq_len
pred_len = seq_len
plot = True
temp_pred = model.predict(X_te)
plot_values = []
for i in X_te[0]:
    plot_values.append((1+i[0])*float(orig_data[0]))
plot_values.append((1+temp_pred[0][0])*float(orig_data[0]))
pprint.pprint(plot_values)
# if plot:
#     # pred = helper.predict_seq_mul(model, X_te, win_size, pred_len)
#     print("I have now predicted the next 50 days of the S&P 500 index. Here are the results starting with the value of X-te and then the prediction: ")
#     pprint.pprint(X_te)
#     pprint.pprint(temp_pred)           
#     helper.plot_mul(temp_pred, X_te, pred_len)
temp_pred = model.predict(X_te)
if plot:
    pred = helper.predict_seq_mul(model, X_te, win_size, pred_len)
    print("I have now predicted the next 50 days of the S&P 500 index. Here are the results starting with the value of X-te and then the prediction: ")
    pprint.pprint(X_te)
    pprint.pprint(pred)           
    helper.plot_mul(pred, X_te, pred_len)
