import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder


def load_data():
    # 修改这里以加载附件2
    data_sc = pd.read_csv("附件2.csv", encoding="gb2312")
    data_routes = pd.read_csv("附件3.csv", encoding="gb2312")
    data_changes = pd.read_csv("附件4.csv", encoding="gb2312")
    return data_sc, data_routes, data_changes



def preprocess_data(data_sc, data_changes):
    # Combining date and hour into a single datetime column
    data_sc["完整时间"] = pd.to_datetime(
        data_sc["日期"].astype(str) + " " + data_sc["小时"].astype(str) + ":00"
    )

    # Adding new time-related features
    data_sc["小时"] = data_sc["完整时间"].dt.hour  # Hour of the day
    data_sc["星期几"] = data_sc["完整时间"].dt.weekday  # Day of the week (Monday=0, Sunday=6)
    data_sc["月份"] = data_sc["完整时间"].dt.month  # Month of the year

    # One-Hot Encoding for sorting centers
    ohe = OneHotEncoder()
    encoded_centers = ohe.fit_transform(data_sc[["分拣中心"]].astype(str)).toarray()
    center_df = pd.DataFrame(encoded_centers, columns=ohe.get_feature_names_out())
    data_sc = pd.concat([data_sc, center_df], axis=1)

    return data_sc, ohe


def predict_future_volume(data_sc, ohe, data_changes):
    # Generating data for the future date range within each hour
    last_time = data_sc["完整时间"].max()
    future_dates = pd.date_range(last_time + pd.Timedelta(hours=1), periods=744, freq="h")
    future_data = pd.DataFrame({"完整时间": future_dates})
    future_data["小时"] = future_data["完整时间"].dt.hour
    future_data["星期几"] = future_data["完整时间"].dt.weekday
    future_data["月份"] = future_data["完整时间"].dt.month

    # Predicting future volume for each sorting center
    future_volumes = []
    centers = ohe.get_feature_names_out()
    for center in centers:
        center_data = data_sc[data_sc[center] == 1]
        if not center_data.empty:
            # Train a model to predict volume
            X = center_data[["小时", "星期几", "月份"]].values
            y = center_data["货量"].values
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Use the model to predict future volume
            future_X = future_data[["小时", "星期几", "月份"]].values
            future_y = model.predict(future_X)
            for date, volume in zip(future_dates, future_y):
                cleaned_center_name = center.replace("x0_分拣中心_", "")
                future_volumes.append(
                    [cleaned_center_name, date.strftime("%Y/%m/%d %H:%M"), volume]
                )

    future_volumes_df = pd.DataFrame(
        future_volumes, columns=["分拣中心", "日期时间", "货量"]
    )

    # Considering changes in routing paths from Attachment 4
    future_volumes_df.set_index(["分拣中心", "日期时间"], inplace=True)
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
    future_volumes_df.to_csv("predicted_future_volumes_hours.csv", index=False)
    print("预测完成，结果已保存到 'predicted_future_volumes_hours.csv'")


if __name__ == "__main__":
    main()
