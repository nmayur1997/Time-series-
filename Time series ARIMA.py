#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yfinance


# In[6]:


import yfinance as yf


# In[7]:


df=yf.download('BTC-USD')


# In[8]:


df


# In[5]:


pip install numpy pandas matplotlib 


# In[6]:


pip install statsmodels


# In[ ]:





# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Assuming df is your DataFrame with a 'Date' index and 'Adj Close' column
plt.plot(df.index, df['Adj Close'])
plt.show()

# Train test split
to_row = int(len(df) * 0.9)
training_data = list(df[0:to_row]['Adj Close'])
testing_data = list(df[to_row:]['Adj Close'])

plt.figure(figsize=(10, 6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df[0:to_row]['Adj Close'], 'green', label='Train data')
plt.plot(df[to_row:]['Adj Close'], 'blue', label='Test data')
plt.legend()

model_predictions = []
errors = []  # To store differences between actual and predicted values
n_test_obser = len(testing_data)

for i in range(n_test_obser):
    model = ARIMA(training_data, order=(4, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast(steps=1)
    yhat = output[0]  # Get the first forecasted value
    model_predictions.append(yhat)
    actual_test_value = testing_data[i]
    training_data.append(actual_test_value)
    
    # Calculate absolute error (difference between predicted and actual)
    error = abs(yhat - actual_test_value)
    errors.append(error)

plt.figure(figsize=(15, 9))
plt.grid(True)
date_range = df[to_row:].index
plt.plot(date_range, model_predictions, color='blue', marker='o', linestyle='dashed', label='BTC Predicted Price')
plt.plot(date_range, testing_data, color='red', label='BTC Actual Price')

plt.title('Bitcoin Price Prediction vs Actual')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate Mean Absolute Error (MAE)
mae = np.mean(errors)
print(f'Mean Absolute Error: {mae}')


# In[52]:


from sklearn.ensemble import RandomForestRegressor

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(np.array(training_data).reshape(-1, 1), training_data)  # Fit on training data

# Predict on testing data
rf_predictions = rf_model.predict(np.array(testing_data).reshape(-1, 1))

# Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
rf_mse = mean_squared_error(testing_data, rf_predictions)
rf_mae = mean_absolute_error(testing_data, rf_predictions)

print(f'Random Forest Mean Squared Error (MSE): {rf_mse}')
print(f'Random Forest Mean Absolute Error (MAE): {rf_mae}')


# In[53]:


from sklearn.ensemble import GradientBoostingRegressor

# Train a Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(np.array(training_data).reshape(-1, 1), training_data)  # Fit on training data

# Predict on testing data
gb_predictions = gb_model.predict(np.array(testing_data).reshape(-1, 1))

# Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
gb_mse = mean_squared_error(testing_data, gb_predictions)
gb_mae = mean_absolute_error(testing_data, gb_predictions)

print(f'Gradient Boosting Mean Squared Error (MSE): {gb_mse}')
print(f'Gradient Boosting Mean Absolute Error (MAE): {gb_mae}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




