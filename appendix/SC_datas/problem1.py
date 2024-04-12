import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 假设数据已经按照分拣中心编号和日期排列好
# 每个文件名格式为 'SC{i}.csv'，其中 i 是分拣中心编号
existing_scs = [1,2,3,4,5,6,7,8,9,10,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,43,44,46,47,48,49,51,52,53,54,55,56,57,58,60,61,63,66,68] 

# 加载数据
def load_data(num_centers=68):
    all_data = []
    for i in existing_scs:
        data = pd.read_csv(f'SC{i}.csv')
        data['SC_id'] = i
        all_data.append(data)
    return pd.concat(all_data, ignore_index=True)

# 数据预处理
def preprocess(data):
    # 假设数据包含日期和货量两列，日期列名为 'date'，货量列名为 'value'
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = (pd.to_datetime(data['date'])).dt.year
    data['month'] = (pd.to_datetime(data['date'])).dt.month
    data['day'] = (pd.to_datetime(data['date'])).dt.day
    data['weekday'] = (pd.to_datetime(data['date'])).dt.weekday
    return data

# 训练模型
def train_model(data):
    features = data[['SC_id', 'year', 'month', 'day', 'weekday']]
    target = data['value']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=50)
    
    model = RandomForestRegressor(n_estimators=100, random_state=50)
    model.fit(X_train, y_train)
    
    # 预测和评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    return model

# 预测未来的货量
def predict_future(model, start_date, num_days, num_centers):
    future_dates = pd.date_range(start_date, periods=num_days)
    future_data = pd.DataFrame({
        'date': np.repeat(future_dates, len(existing_scs)),
        'SC_id': np.tile(existing_scs, num_days)
    })
    future_data = preprocess(future_data)
    features = future_data[['SC_id', 'year', 'month', 'day', 'weekday']]
    predictions = model.predict(features)
    future_data['predicted_volume'] = predictions
    return future_data

# 保存结果到CSV
def save_predictions_to_csv(predictions, file_name):
    predictions['date'] = predictions['date'].dt.strftime('%Y/%m/%d')
    predictions.to_csv(file_name, index=False)
    print(f'Saved predictions to {file_name}')

# 主函数
def main():
    data = load_data()
    data = preprocess(data)
    model = train_model(data)
    future_predictions = predict_future(model, '2023-12-01', 30, 68)
    save_predictions_to_csv(future_predictions, 'predicted_volumes.csv')

if __name__ == '__main__':
    main()
