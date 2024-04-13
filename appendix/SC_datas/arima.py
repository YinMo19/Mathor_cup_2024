import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# 假设存在的SCid列表
data_for_sc = pd.read_csv("../../附件/附件1.csv", encoding="GB2312")
ALL_SC = list(set(data_for_sc["分拣中心"]))
existing_scs = list(map(lambda SC_: int(SC_[2:]), ALL_SC))
existing_scs.sort()


# 加载和预处理数据的函数不变
def load_data(existing_scs):
    all_data = []
    for i in existing_scs:
        data = pd.read_csv(f"SC{i}.csv")
        data["SC_id"] = i
        all_data.append(data)
    return pd.concat(all_data, ignore_index=True)


def preprocess(data):
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(by=["SC_id", "date"])
    return data

def preprocess(data):
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(by=["SC_id", "date"])
    return data

# 训练 ARIMA 模型
def train_sarima_models(data):
    models = {}
    for sc_id, group in data.groupby("SC_id"):
        # 需要先确定最佳的SARIMA模型参数
        # 这里使用了(1, 1, 1)x(1, 1, 1, 12)作为示例，实际上你需要通过模型选择过程来确定最佳参数
        sarima_order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)
        model = SARIMAX(
            group["value"],
            order=sarima_order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted_model = model.fit(disp=False)
        # 评估模型可以使用滚动预测的方式，这里暂时省略
        models[sc_id] = fitted_model
    return models


# 预测未来的货量
def predict_future(models, start_date, num_days, existing_scs):
    future_dates = pd.date_range(start_date, periods=num_days)
    all_predictions = []
    for sc_id, model in models.items():
        predictions = model.forecast(steps=num_days)
        sc_predictions = pd.DataFrame(
            {"date": future_dates, "SC_id": sc_id, "predicted_volume": predictions}
        )
        all_predictions.append(sc_predictions)
    return pd.concat(all_predictions, ignore_index=True)


# 保存结果到CSV的函数不变
def save_predictions_to_csv(predictions, file_name):
    predictions["date"] = predictions["date"].dt.strftime("%Y-%m-%d")
    predictions.to_csv(file_name, index=False)
    print(f"Saved predictions to {file_name}")


# 主函数
# 主函数
def main():
    data = load_data(existing_scs)
    data = preprocess(data)
    models = train_sarima_models(data)
    future_predictions = predict_future(models, "2023-08-01", 153, existing_scs)
    save_predictions_to_csv(future_predictions, "predicted_volumes.csv")


if __name__ == "__main__":
    main()
