import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import numpy as np

# 假设存在的SCid列表
existing_scs = [1,2,3,4,5,6,7,8,9,10,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,43,44,46,47,48,49,51,52,53,54,55,56,57,58,60,61,63,66,68] 

# 加载数据
def load_data(existing_scs):
    all_data = {}
    for sc_id in existing_scs:
        try:
            data = pd.read_csv(f'SC_{sc_id}.csv', parse_dates=['date'], index_col='date')
            all_data[sc_id] = data
        except FileNotFoundError:
            print(f"File for center {sc_id} not found.")
    return all_data

# 数据清洗和预处理
def preprocess_data(data):
    data = data.asfreq('D')  # 确保数据频率为每天
    data = data.fillna(method='ffill')  # 填充缺失值
    return data

# 训练ARIMA模型
def train_arima(data):
    model = auto_arima(data, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    return model

# 预测未来的货量
def predict_future(model, periods):
    forecast = model.predict(n_periods=periods)
    return forecast

# 主函数
def main():
    data_dict = load_data(existing_scs)
    predictions = {}
    
    for sc_id, data in data_dict.items():
        print(f"Processing center {sc_id}...")
        data = preprocess_data(data)
        model = train_arima(data['value'])
        future_forecast = predict_future(model, 30)  # 预测未来30天
        
        predictions[sc_id] = future_forecast
        print(f"Predictions for center {sc_id}: {future_forecast}")

if __name__ == '__main__':
    main()
