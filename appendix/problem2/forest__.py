import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error


def load_data():
    # 加载数据
    data_sc = pd.read_csv("附件1.csv", encoding="gb2312")
    data_routes = pd.read_csv("附件3.csv", encoding="gb2312")
    data_changes = pd.read_csv("附件4.csv", encoding="gb2312")
    return data_sc, data_routes, data_changes


def preprocess_data(data_sc, data_changes):
    # Convert '日期' to datetime and extract more features
    data_sc["日期"] = pd.to_datetime(data_sc["日期"], format="%Y/%m/%d")
    data_sc["月份"] = data_sc["日期"].dt.month  # Month as a feature
    data_sc["星期几"] = data_sc["日期"].dt.weekday  # Day of the week as a feature
    data_sc["年中日"] = data_sc["日期"].dt.dayofyear  # Day of the year as a feature

    # One-Hot Encoding for sorting centers
    ohe = OneHotEncoder()
    encoded_centers = ohe.fit_transform(data_sc[["分拣中心"]].astype(str)).toarray()
    center_df = pd.DataFrame(encoded_centers, columns=ohe.get_feature_names_out())
    data_sc = pd.concat([data_sc, center_df], axis=1)

    return data_sc, ohe


def predict_future_volume(data_sc, ohe, data_changes):
    # Generate future dates starting 120 days after the minimum date in the data
    future_dates = pd.date_range(start=data_sc["日期"].max() + pd.Timedelta(days=1), periods=31, freq="D")
    future_data = pd.DataFrame({"日期": future_dates})
    future_data["月份"] = future_data["日期"].dt.month
    future_data["星期几"] = future_data["日期"].dt.weekday
    future_data["年中日"] = future_data["日期"].dt.dayofyear

    # Predict future volume for each sorting center
    future_volumes = []
    centers = ohe.get_feature_names_out()
    for center in centers:
        center_data = data_sc[data_sc[center] == 1]
        if not center_data.empty:
            # Train a model to predict volume
            X = center_data[["年中日", "月份", "星期几"]]
            y = center_data["货量"]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Predict future volume
            future_X = future_data[["年中日", "月份", "星期几"]]
            future_y = model.predict(future_X)
            for date, volume in zip(future_dates, future_y):
                cleaned_center_name = center.replace("x0_分拣中心_", "")
                future_volumes.append(
                    [cleaned_center_name, date.strftime("%Y/%m/%d"), volume]
                )

    future_volumes_df = pd.DataFrame(
        future_volumes, columns=["分拣中心", "日期", "货量"]
    )

    # Adjust predicted volumes based on changes from Attachment 4
    future_volumes_df.set_index(["分拣中心", "日期"], inplace=True)
    for index, row in data_changes.iterrows():
        from_center = "x0_分拣中心_" + row["始发分拣中心"]
        to_center = "分拣中心_" + row["到达分拣中心"]
        if from_center in centers:
            future_volumes_df.loc[(to_center,), "货量"] += (
                future_volumes_df.loc[(from_center,), "货量"] * 0.1
            )
            future_volumes_df.loc[(from_center,), "货量"] *= 0.9

    future_volumes_df.reset_index(inplace=True)
    return future_volumes_df


def main():
    data_sc, data_routes, data_changes = load_data()
    data_sc, ohe = preprocess_data(data_sc, data_changes)
    future_volumes_df = predict_future_volume(data_sc, ohe, data_changes)
    future_volumes_df.to_csv("predicted_future_volumes.csv", index=False)
    print("预测完成，结果已保存到 'predicted_future_volumes.csv'")


if __name__ == "__main__":
    main()
