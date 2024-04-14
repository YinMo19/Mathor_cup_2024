import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 数据加载和预处理
def load_and_preprocess_data():
    data_sc = pd.read_csv('附件1.csv', encoding='gb2312')
    data_routes = pd.read_csv('附件3.csv', encoding='gb2312')
    data_changes = pd.read_csv('/mnt/data/附件4.csv', encoding='gb2312')

    data_sc['日期'] = pd.to_datetime(data_sc['日期'], format='%Y/%m/%d')
    data_sc.sort_values('日期', inplace=True)

    le = LabelEncoder()
    data_sc['分拣中心'] = le.fit_transform(data_sc['分拣中心'])

    return data_sc, le

def create_dataset(data, look_back=14):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future(model, data, le, look_back=14, days=30):
    last_batch = data['货量'].values[-look_back:]
    last_batch = last_batch.reshape((1, look_back, 1))
    predictions = []
    for _ in range(days):
        pred = model.predict(last_batch)[0][0]
        predictions.append(pred)
        last_batch = np.append(last_batch[:, 1:, :], [[pred]], axis=1)
    return predictions

def main():
    data_sc, le = load_and_preprocess_data()
    scaler = MinMaxScaler()
    data_sc['货量'] = scaler.fit_transform(data_sc[['货量']])

    X, y = create_dataset(data_sc['货量'], look_back=14)
    X = X.reshape(X.shape[0], 14, 1)

    model = build_model((14, 1))
    model.fit(X, y, epochs=50, verbose=1)

    future_values = predict_future(model, data_sc, le)
    future_values = scaler.inverse_transform(np.array(future_values).reshape(-1, 1))

    future_dates = pd.date_range(start=data_sc['日期'].max() + pd.Timedelta(days=1), periods=30)
    results_df = pd.DataFrame({
        'date': future_dates,
        'value': future_values.flatten()
    })
    results_df.to_csv('future_predictions.csv', index=False)
    print("未来预测结果已保存到 'future_predictions.csv'")

if __name__ == '__main__':
    main()
