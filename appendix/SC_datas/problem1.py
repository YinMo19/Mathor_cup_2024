import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 假设数据已经按照分拣中心编号和日期排列好
# 每个文件名格式为 'SC{i}.csv'，其中 i 是分拣中心编号

listSC=['SC58', 'SC4', 'SC52', 'SC10', 'SC28', 'SC3', 'SC18', 'SC35', 'SC25', 'SC9', 'SC43', 'SC19', 'SC47', 'SC1', 'SC14', 'SC5', 'SC44', 'SC61', 'SC63', 'SC46', 'SC2', 'SC20', 'SC55', 'SC60', 'SC24', 'SC68', 'SC66', 'SC34', 'SC37', 'SC6', 'SC26', 'SC36', 'SC21', 'SC57', 'SC27', 'SC41', 'SC39', 'SC15', 'SC32', 'SC23', 'SC17', 'SC56', 'SC12', 'SC30', 'SC7', 'SC8', 'SC29', 'SC48', 'SC40', 'SC22', 'SC54', 'SC16', 'SC51', 'SC49', 'SC31', 'SC38', 'SC53']
# 加载数据
def load_data(num_centers=1):
    all_data = []
    for i in range(1, num_centers + 1):
        if( f'SC{i}' not in listSC):
            continue
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
        'date': np.repeat(future_dates, num_centers),
        'SC_id': np.tile(range(1, num_centers + 1), num_days)
    })
    future_data = preprocess(future_data)
    features = future_data[['SC_id', 'year', 'month', 'day', 'weekday']]
    predictions = model.predict(features)
    future_data['predicted_volume'] = predictions
    return future_data

# 主函数
def main():
    data = load_data()
    data = preprocess(data)
    model = train_model(data)
    future_predictions = predict_future(model, '2023-12-01', 30, 1)
    print(future_predictions)

if __name__ == '__main__':
    main()
