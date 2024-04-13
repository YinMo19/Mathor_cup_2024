import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 假设存在的SCid列表
data_for_sc = pd.read_csv("../../附件/附件1.csv", encoding="GB2312")
ALL_SC = list(set(data_for_sc["分拣中心"]))
existing_scs = list(map(lambda SC_: int(SC_[2:]), ALL_SC))
existing_scs.sort()


# 加载数据
def load_data(existing_scs):
    all_data = []
    for sc_id in existing_scs:
        try:
            data = pd.read_csv(f"SC{sc_id}.csv")
            data["center_id"] = sc_id
            all_data.append(data)
        except FileNotFoundError:
            print(f"File for center {sc_id} not found.")
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()


# 数据清洗和预处理
def preprocess_data(data):
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data.dropna(subset=["date", "value"], inplace=True)

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["weekday"] = data["date"].dt.weekday

    scaler = StandardScaler()
    data[["year", "month", "day", "weekday"]] = scaler.fit_transform(
        data[["year", "month", "day", "weekday"]]
    )

    return data


# 模型训练和参数调整
def train_and_optimize_model(X_train, y_train):
    param_grid = {
        "n_estimators": [100 * i for i in range(1, 10)],
        "max_depth": [10 * i for i in range(1, 10)],
        "min_samples_split": [10 * i for i in range(1, 10)],
    }
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="neg_mean_squared_error", verbose=2
    )
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_estimator_


if __name__ == "__main__":
    data = load_data(existing_scs)
    if not data.empty:
        data = preprocess_data(data)
        features = data[["center_id", "year", "month", "day", "weekday"]]
        target = data["value"]

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        best_model = train_and_optimize_model(X_train, y_train)

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE: {mse}")
    else:
        print("No data loaded, please check the data files.")
