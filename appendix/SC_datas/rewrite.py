import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import numpy as np

# Load Data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data = data.asfreq('D')
    data.fillna(method='ffill', inplace=True)
    return data

# Check Stationarity and Perform Necessary Transformations
def preprocess_data(data):
    data['log_volume'] = np.log(data['volume'])
    return data

# Auto ARIMA to identify best parameters
def auto_arima_model(data):
    seasonal = True  # Assuming seasonality might be involved
    model = auto_arima(data['log_volume'], seasonal=seasonal, m=7, trace=True,
                       error_action='ignore', suppress_warnings=True,
                       stepwise=True)
    print(model.summary())
    return model

# Fit SARIMAX Model
def fit_model(data, order, seasonal_order):
    model = SARIMAX(data['log_volume'], order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    print(results.summary())
    return results

# Forecast
def forecast(model, periods=30):
    forecast = model.get_forecast(steps=periods)
    mean_forecast = np.exp(forecast.predicted_mean)
    return mean_forecast

# Load and preprocess data
data = load_data('path_to_your_data.csv')
data = preprocess_data(data)

# Model Identification
model = auto_arima_model(data)

# Fit model
order = model.order
seasonal_order = model.seasonal_order
results = fit_model(data, order, seasonal_order)

# Forecast
future_values = forecast(results)

# Save forecasted values
future_values.to_csv('forecast_output.csv', header=True)

# Diagnostic plots
results.plot_diagnostics(figsize=(15, 12))
plt.show()
