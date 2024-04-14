import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# 假设存在的SCid列表
data_for_sc = pd.read_csv("../../附件/附件1.csv", encoding="GB2312")
ALL_SC = list(set(data_for_sc["分拣中心"]))
existing_scs = list(map(lambda SC_: int(SC_[2:]), ALL_SC))
existing_scs.sort()


# 检查序列平稳性
def check_stationarity(series):
    result = adfuller(series.dropna())
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print("\t%s: %.3f" % (key, value))
    if result[1] > 0.05:
        print("Series is not stationary")
    else:
        print("Series is stationary")


# 加载数据
def load_data(existing_scs):
    all_data = {}
    for sc_id in existing_scs:
        try:
            data = pd.read_csv(f"SC{sc_id}.csv", parse_dates=["date"], index_col="date")
            data = data.asfreq("H")  # 设置数据频率为每小时
            data = data.fillna(method="ffill")  # 填充缺失值
            all_data[sc_id] = data["value"]  # 假设货量列名为 'value'
            print(f"Checking stationarity for center {sc_id}:")
            check_stationarity(data["value"])  # 检查平稳性
        except FileNotFoundError:
            print(f"File for center {sc_id} not found.")
    return all_data


# 训练SARIMA模型并进行预测
def train_and_predict(data, sc_id):
    # 使用auto_arima自动找到最优参数
    model = auto_arima(
        data,
        start_p=1,
        start_q=1,
        max_p=2,
        max_q=2,
        m=24,
        start_P=0,
        max_P=1,
        max_D=1,
        max_Q=1,
        seasonal=True,
        d=1,
        D=1,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        n_jobs=-1,
    )

    print(
        f"Best SARIMA model order {model.order} and seasonal order {model.seasonal_order} for SC_ID {sc_id}"
    )
    model_fit = model.fit(data)
    future_forecast = model.predict(
        n_periods=720
    )  # 预测未来一个月的小时数据，一天24小时*30天
    return future_forecast


# 将预测结果保存到CSV文件
def save_predictions_to_csv(predictions):
    rows = []
    for sc_id, forecast in predictions.items():
        dates = pd.date_range(start="2024/01/01", periods=720, freq="H")
        for date, value in zip(dates, forecast):
            rows.append(
                {
                    "SC_ID": sc_id,
                    "date": date.strftime("%Y/%m/%d %H:%M"),
                    "value": value,
                }
            )

    df = pd.DataFrame(rows, columns=["SC_ID", "date", "value"])
    df.to_csv("output.csv", index=False)
    print("Saved predictions to 'output.csv'")


# 主函数
def main():
    data_dict = load_data(existing_scs)
    predictions = {}

    for sc_id, data in data_dict.items():
        print(f"Training SARIMA for center {sc_id}...")
        forecast = train_and_predict(data, sc_id)
        predictions[sc_id] = forecast

    save_predictions_to_csv(predictions)


if __name__ == "__main__":
    main()
