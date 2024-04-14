import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

def load_data():
    data_sc = pd.read_csv('附件1.csv', encoding='gb2312')
    data_routes = pd.read_csv('附件3.csv', encoding='gb2312')
    data_changes = pd.read_csv('附件4.csv', encoding='gb2312')
    return data_sc, data_routes, data_changes

def preprocess_data(data_sc, data_routes, data_changes):
    data_sc['日期'] = pd.to_datetime(data_sc['日期'], format='%Y/%m/%d')
    le = LabelEncoder()
    data_sc['分拣中心'] = le.fit_transform(data_sc['分拣中心'])

    # 计算每个分拣中心的初始平均货量
    avg_departure = data_routes.groupby('始发分拣中心')['货量'].mean().rename('平均发出货量').reset_index()
    avg_arrival = data_routes.groupby('到达分拣中心')['货量'].mean().rename('平均到达货量').reset_index()

    # 转换分拣中心编码
    avg_departure['始发分拣中心'] = le.transform(avg_departure['始发分拣中心'])
    avg_arrival['到达分拣中心'] = le.transform(avg_arrival['到达分拣中心'])

    # 合并调整后的货量信息
    data_sc = pd.merge(data_sc, avg_departure, left_on='分拣中心', right_on='始发分拣中心', how='left')
    data_sc = pd.merge(data_sc, avg_arrival, left_on='分拣中心', right_on='到达分拣中心', how='left')
    data_sc.fillna({'平均发出货量': 0, '平均到达货量': 0}, inplace=True)

    return data_sc, le

def create_features_labels(data):
    data['year'] = data['日期'].dt.year
    data['month'] = data['日期'].dt.month
    data['day'] = data['日期'].dt.day
    data['weekday'] = data['日期'].dt.weekday
    features = data[['分拣中心', 'year', 'month', 'day', 'weekday', '平均发出货量', '平均到达货量']]
    labels = data['货量']
    return features, labels

def train_and_predict(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    return model

def predict_future(model, last_date, le, data_sc):
    # 生成未来30天的日期
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    future_df = pd.DataFrame({
        'date': pd.to_datetime(list(future_dates) * len(data_sc['分拣中心'].unique())),
        '分拣中心': list(data_sc['分拣中心'].unique()) * len(future_dates)
    })
    
    future_df['year'] = future_df['date'].dt.year
    future_df['month'] = future_df['date'].dt.month
    future_df['day'] = future_df['date'].dt.day
    future_df['weekday'] = future_df['date'].dt.weekday

    # 估计平均发出货量和平均到达货量
    avg_departure = data_sc['平均发出货量'].mean()
    avg_arrival = data_sc['平均到达货量'].mean()

    future_df['平均发出货量'] = avg_departure
    future_df['平均到达货量'] = avg_arrival

    # 提取特征
    features = future_df[['分拣中心', 'year', 'month', 'day', 'weekday', '平均发出货量', '平均到达货量']]
    
    # 使用模型进行预测
    predicted_values = model.predict(features)
    future_df['value'] = predicted_values
    future_df['date'] = future_df['date'].dt.strftime('%Y/%m/%d')  # 格式化日期

    results_df = future_df[['分拣中心', 'date', 'value']]
    results_df.rename(columns={'分拣中心': 'SC_ID'}, inplace=True)

    # 保存到 CSV
    results_df.to_csv('future_predictions.csv', index=False)
    print("未来预测结果已保存到 'future_predictions.csv'")



def main():
    data_sc, data_routes, data_changes = load_data()
    data_sc, le = preprocess_data(data_sc, data_routes, data_changes)
    features, labels = create_features_labels(data_sc)
    model = train_and_predict(features, labels)
    last_date = data_sc['日期'].max()  # 获取已知数据的最后一个日期
    predict_future(model, last_date, le, data_sc)

if __name__ == "__main__":
    main()
