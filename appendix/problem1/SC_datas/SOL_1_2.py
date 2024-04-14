import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 假设数据已经按照分拣中心编号和日期排列好
existing_scs = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    43,
    44,
    46,
    47,
    48,
    49,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    60,
    61,
    63,
    66,
    68,
]


# 加载数据
def load_data(existing_scs):
    all_data = []
    for i in existing_scs:
        data = pd.read_csv(f"SC{i}.csv")
        data["SC_id"] = i
        all_data.append(data)
    return pd.concat(all_data, ignore_index=True)


# 数据预处理
def preprocess(data):
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["weekday"] = data["date"].dt.weekday
    return data


# 训练 ARIMA 模型
def train_arima(data):
    model = ARIMA(data["value"], order=(1, 1, 1))
    model_fit = model.fit()
    return model_fit


# 训练 LSTM 模型
def train_lstm(features, target):
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    features_scaled = features_scaled.reshape(
        features_scaled.shape[0], 1, features_scaled.shape[1]
    )
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(1, features.shape[1])))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(features_scaled, target, epochs=50, batch_size=10, verbose=0)
    return model, scaler


# 主函数
def main():
    data = load_data(existing_scs)
    data = preprocess(data)
    features = data[["year", "month", "day", "weekday"]]
    target = data["value"]

    # ARIMA
    arima_model = train_arima(data)

    # LSTM
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=50
    )
    lstm_model, scaler = train_lstm(X_train, y_train)

    # 评估 LSTM
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = X_test_scaled.reshape(
        X_test_scaled.shape[0], 1, X_test_scaled.shape[1]
    )
    y_pred = lstm_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"LSTM Mean Squared Error: {mse}")


if __name__ == "__main__":
    main()
