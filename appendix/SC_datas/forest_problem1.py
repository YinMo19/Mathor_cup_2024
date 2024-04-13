import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 假设数据已经按照分拣中心编号和日期排列好
# 每个文件名格式为 'SC{i}.csv'，其中 i 是分拣中心编号
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
    # 假设数据包含日期和货量两列，日期列名为 'date'，货量列名为 'value'
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = (pd.to_datetime(data["date"])).dt.year
    data["month"] = (pd.to_datetime(data["date"])).dt.month
    data["day"] = (pd.to_datetime(data["date"])).dt.day
    data["weekday"] = (pd.to_datetime(data["date"])).dt.weekday
    return data


def train_stacking_model(data):
    # 分割数据为训练集和测试集
    features = data[["SC_id", "year", "month", "day", "weekday"]]
    target = data["value"]
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=50,
    )

    # 排除 "SC_id"，只对其他数值特征进行归一化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[["year", "month", "day", "weekday"]])
    X_test_scaled = scaler.transform(X_test[["year", "month", "day", "weekday"]])

    # LSTM 需要三维输入 [samples, timesteps, features]
    X_train_scaled = X_train_scaled.reshape(
        (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    )
    X_test_scaled = X_test_scaled.reshape(
        (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    )

    # 训练随机森林模型
    rf_model = RandomForestRegressor(
        n_estimators=300, random_state=42, min_samples_split=20
    )
    rf_model.fit(
        X_train[["year", "month", "day", "weekday"]], y_train
    )  # 注意排除了 "SC_id"

    # 配置 LSTM 模型
    lstm_model = Sequential()
    lstm_model.add(
        LSTM(units=50, return_sequences=True, input_shape=(1, 4))
    )  # 这里的 input_shape 对应排除 "SC_id" 后的特征数量
    lstm_model.add(LSTM(units=50, return_sequences=True))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer="adam", loss="mean_squared_error")

    # 训练 LSTM 模型
    lstm_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32)

    # 使用模型进行预测
    rf_predictions = rf_model.predict(X_test[["year", "month", "day", "weekday"]])
    lstm_predictions = lstm_model.predict(X_test_scaled).flatten()

    # 集成学习
    stacked_predictions = np.column_stack((rf_predictions, lstm_predictions))
    meta_model = LinearRegression()
    meta_model.fit(stacked_predictions, y_test)  # 使用真实目标值 y_test 进行训练

    # 预测和评估
    rf_mse = mean_squared_error(y_test, rf_predictions)
    lstm_mse = mean_squared_error(y_test, lstm_predictions)
    stacked_mse = mean_squared_error(y_test, meta_model.predict(stacked_predictions))

    print(f"Random Forest Mean Squared Error: {rf_mse}")
    print(f"LSTM Mean Squared Error: {lstm_mse}")
    print(f"Stacking Mean Squared Error: {stacked_mse}")

    return rf_model, lstm_model, meta_model, scaler


# 预测未来的货量
def predict_future(models, scaler, start_date, num_days, existing_scs):

    future_dates = pd.date_range(start_date, periods=num_days)
    future_data = pd.DataFrame(
        {
            "date": np.repeat(future_dates, len(existing_scs)),
            "SC_id": np.tile(existing_scs, num_days),
        }
    )

    # 预处理数据，添加年、月、日、星期等特征
    future_data["year"] = future_data["date"].dt.year
    future_data["month"] = future_data["date"].dt.month
    future_data["day"] = future_data["date"].dt.day
    future_data["weekday"] = future_data["date"].dt.weekday

    # 这里我们预处理完数据后，需要进行相同的归一化处理
    features = future_data[["year", "month", "day", "weekday"]]
    features_scaled = scaler.transform(features)  # 使用与训练数据相同的scaler

    # 使用元模型进行预测，这里假设 models 包含随机森林、LSTM 和元模型
    rf_model, lstm_model, meta_model = models
    rf_predictions = rf_model.predict(features)
    lstm_predictions = lstm_model.predict(
        features_scaled.reshape(features_scaled.shape[0], 1, features_scaled.shape[1])
    ).flatten()
    stacked_predictions = np.column_stack((rf_predictions, lstm_predictions))
    final_predictions = meta_model.predict(stacked_predictions)

    future_data["predicted_volume"] = final_predictions
    return future_data


# 保存结果到CSV
def save_predictions_to_csv(predictions, file_name):
    predictions["date"] = predictions["date"].dt.strftime("%Y/%m/%d")
    predictions.to_csv(file_name, index=False)
    print(f"Saved predictions to {file_name}")


# 主函数
def main():
    data = load_data(existing_scs)
    data = preprocess(data)
    rf_model, lstm_model, meta_model, scaler = train_stacking_model(data)
    future_predictions = predict_future(
        (rf_model, lstm_model, meta_model), scaler, "2023-08-01", 153, existing_scs
    )
    save_predictions_to_csv(future_predictions, "predicted_volumes.csv")


if __name__ == "__main__":
    main()
