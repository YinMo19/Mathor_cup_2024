import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# 假设存在的SCid列表
existing_scs = [1,2,3,4,5,6,7,8,9,10,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,43,44,46,47,48,49,51,52,53,54,55,56,57,58,60,61,63,66,68] 

# 加载数据
def load_data(existing_scs):
    all_data = {}
    for sc_id in existing_scs:
        try:
            # 假设CSV文件名为 'center_{sc_id}.csv'，日期列名为 'date'，货量列名为 'volume'
            data = pd.read_csv(f'SC{sc_id}.csv', parse_dates=['date'], index_col='date')
            data = data.asfreq('D')  # 确保数据频率为每天
            data = data.fillna(method='ffill')  # 填充缺失值
            all_data[sc_id] = data['value']  # 假设货量列名为 'volume'
        except FileNotFoundError:
            print(f"File for center {sc_id} not found.")
    return all_data

# 训练ARIMA模型并进行预测
def train_and_predict(data, sc_id):
    model = ARIMA(data, order=(4, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)  # 预测未来30天
    return forecast

# 将预测结果保存到CSV文件
def save_predictions_to_csv(predictions):
    # 创建列表以存储每行数据
    rows = []
    for sc_id, forecast in predictions.items():
        dates = pd.date_range(start=forecast.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        for date, value in zip(dates, forecast):
            rows.append({
                'SC_ID': sc_id,
                'date': date.strftime('%Y/%m/%d'),  # 格式化日期为YYYY/MM/DD
                'value': value
            })
    
    # 创建DataFrame
    df = pd.DataFrame(rows, columns=['SC_ID', 'date', 'value'])
    # 保存到CSV
    df.to_csv('output.csv', index=False)
    print("Saved predictions to 'output.csv'")

# 主函数
def main():
    data_dict = load_data(existing_scs)
    predictions = {}

    for sc_id, data in data_dict.items():
        print(f"Processing center {sc_id}...")
        forecast = train_and_predict(data, sc_id)
        predictions[sc_id] = forecast
        print(f"Predictions for center {sc_id}: {forecast}")

    save_predictions_to_csv(predictions)

if __name__ == '__main__':
    main()