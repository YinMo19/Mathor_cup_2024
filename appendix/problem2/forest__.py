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
    # 预处理数据，准备特征和标签
    data_sc["日期"] = pd.to_datetime(data_sc["日期"], format="%Y/%m/%d")

    # One-Hot Encoding 分拣中心
    ohe = OneHotEncoder()
    encoded_centers = ohe.fit_transform(data_sc[["分拣中心"]].astype(str)).toarray()
    center_df = pd.DataFrame(encoded_centers, columns=ohe.get_feature_names_out())
    data_sc = pd.concat([data_sc, center_df], axis=1)

    return data_sc, ohe


def predict_future_volume(data_sc, ohe, data_changes):
    # 生成未来日期范围内的数据
    future_dates = pd.date_range(data_sc["日期"].min(), periods=31, freq="D")
    future_data = pd.DataFrame({"日期": future_dates})

    # 预测每个分拣中心的未来货量
    future_volumes = []
    centers = ohe.get_feature_names_out()
    for center in centers:
        # 过滤数据到当前分拣中心
        center_data = data_sc[data_sc[center] == 1]
        if not center_data.empty:
            # 训练一个模型来预测货量
            X = (
                center_data[["日期"]]
                .apply(lambda x: x.dt.dayofyear)
                .values.reshape(-1, 1)
            )
            y = center_data["货量"].values
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            # 使用模型预测未来的货量
            future_X = (
                future_data["日期"].apply(lambda x: x.dayofyear).values.reshape(-1, 1)
            )
            future_y = model.predict(future_X)
            for date, volume in zip(future_dates, future_y):
                # 修改这里的处理方式，移除“分拣中心_”和前缀
                cleaned_center_name = center.replace("x0_分拣中心_", "")
                future_volumes.append(
                    [cleaned_center_name, date.strftime("%Y/%m/%d"), volume]
                )

    future_volumes_df = pd.DataFrame(
        future_volumes, columns=["分拣中心", "日期", "货量"]
    )

    # 考虑附件4的流通路径变更
    future_volumes_df.set_index(["分拣中心", "日期"], inplace=True)
    for index, row in data_changes.iterrows():
        from_center = "x0_分拣中心_" + row["始发分拣中心"]
        to_center = "分拣中心_" + row["到达分拣中心"]
        if from_center in centers:
            # 分配部分货量到新的目的地
            future_volumes_df.loc[(to_center,), "货量"] += (
                future_volumes_df.loc[(from_center,), "货量"] * 0.1
            )
            future_volumes_df.loc[(from_center,), "货量"] *= 0.9

    # 重置索引以符合输出格式
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
