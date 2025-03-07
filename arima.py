import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

concatenated_df = pd.read_csv('hourlydata.csv')
concatenated_df.head()

X = concatenated_df['CPU usage [%]']
size = int(len(X) * 0.66)
train, test = X[0:size].reset_index(drop=True), X[size:len(X)].reset_index(drop=True)
history = [x for x in train]
predictions = list()

# Training and predicting with ARIMA model
for t in range(len(test)):
    model = sm.tsa.arima.ARIMA(history, order=(10,0,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

mse = mean_squared_error(test, predictions)
mae = mean_absolute_error(test, predictions)
rmse = sqrt(mse)
r2 = r2_score(test, predictions)
ape = np.abs((np.array(predictions) - np.array(test)) / np.array(test)) * 100
mape = np.mean(ape)
print('Test MSE: %.3f' % mse)
print('Test MAE: %.3f' % mae)
print('Test MAPE: %.3f' % mape)
print('Test RMSE: %.3f' % rmse)
print('Test R2 score: %.3f' % r2)

sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))
plt.plot(test, label='Actual', color='b')
plt.plot(predictions, label='Predicted', color='r')
plt.legend(loc='upper right')
plt.title('ARIMA: Actual vs Predicted CPU usage (MHz)')
plt.ylabel('CPU usage (MHz)')
plt.xlabel('Steps')
plt.grid(True)
plt.savefig('arima_prediction.png')
plt.show()