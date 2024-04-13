import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import matplotlib.pyplot as plt

# 假设存在的SCid列表
existing_scs = [1,2,3,4,5,6,7,8,9,10,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,43,44,46,47,48,49,51,52,53,54,55,56,57,58,60,61,63,66,68] 

# 加载数据
def load_data(sc_id):
    try:
        data = pd.read_csv(f'SC{sc_id}.csv', parse_dates=['date'], index_col='date')
        data = data.asfreq('D')  # 确保数据频率为每天
        data.fillna(method='ffill', inplace=True)  # 填充缺失值
        return data
    except FileNotFoundError:
        print(f"Data file for SC_ID {sc_id} not found.")
        return None

# 预处理数据
def preprocess_data(data):
    # 你可以在这里添加更多的预处理步骤
    data['log_volume'] = np.log(data['value'])
    return data

# 自动ARIMA模型
def auto_arima_model(data):
    model = auto_arima(data['log_volume'], trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    return model

# 训练SARIMAX模型
def fit_model(data, order):
    model = SARIMAX(data['log_volume'], order=order, seasonal_order=(0,0,0,0),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    return results

# 预测未来值
def forecast(model, periods=153):
    forecast = model.get_forecast(steps=periods)
    mean_forecast = np.exp(forecast.predicted_mean)
    return mean_forecast

# 主程序
def main():
    results = []
    for sc_id in sc_ids:
        data = load_data(sc_id)
        if data is not None:
            data = preprocess_data(data)
            model = auto_arima_model(data)
            results_model = fit_model(data, model.order)
            forecast_values = forecast(results_model)
            forecast_values = forecast_values.reset_index(drop=True)
            dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=153, freq='D')
            df = pd.DataFrame({'SC_ID': sc_id, 'Date': dates, 'Forecast': forecast_values})
            results.append(df)
            
    if results:
        final_results = pd.concat(results)
        final_results.to_csv('final_forecast.csv', index=False)
        print("Saved all forecasts to 'final_forecast.csv'.")

if __name__ == '__main__':
    main()
