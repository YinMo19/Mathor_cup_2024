import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error

def load_data():
    data_sc = pd.read_csv("附件1.csv", encoding="gb2312")
    data_routes = pd.read_csv("附件3.csv", encoding="gb2312")
    data_changes = pd.read_csv("附件4.csv", encoding="gb2312")
    return data_sc, data_routes, data_changes

def preprocess_data(data_sc, data_routes):
    data_sc["日期"] = pd.to_datetime(data_sc["日期"], format="%Y/%m/%d")
    le = LabelEncoder()
    data_sc["分拣中心"] = le.fit_transform(data_sc["分拣中心"])

    # 对 data_routes 中的 '始发分拣中心' 进行相同的标签编码
    data_routes['始发分拣中心'] = le.transform(data_routes['始发分拣中心'])

    # One-Hot Encoding for origin and destination centers
    ohe_origin = OneHotEncoder()
    ohe_destination = OneHotEncoder()
    origin_encoded = ohe_origin.fit_transform(data_routes[['始发分拣中心']].astype(str)).toarray()
    destination_encoded = ohe_destination.fit_transform(data_routes[['到达分拣中心']].astype(str)).toarray()

    # Create DataFrames from the encoded arrays
    origin_df = pd.DataFrame(origin_encoded, columns=ohe_origin.get_feature_names_out())
    destination_df = pd.DataFrame(destination_encoded, columns=ohe_destination.get_feature_names_out())

    # Add encoded data back to data_routes
    data_routes = pd.concat([data_routes.drop(['始发分拣中心', '到达分拣中心'], axis=1), origin_df, destination_df], axis=1)

    # Merge the adjusted route information back to data_sc
    data_sc = pd.merge(data_sc, data_routes, left_on="分拣中心", right_index=True, how="left")
    data_sc.fillna(0, inplace=True)  # Fill NaN with zeros where no match is found

    return data_sc, le

def create_features_labels(data):
    data['year'] = data['日期'].dt.year
    data['month'] = data['日期'].dt.month
    data['day'] = data['日期'].dt.day
    data['weekday'] = data['日期'].dt.weekday
    # Include only the necessary columns and the newly added one-hot encoded columns
    feature_columns = [col for col in data.columns if '始发分拣中心' in col or '到达分拣中心' in col]
    features = data[['分拣中心', 'year', 'month', 'day', 'weekday'] + feature_columns]
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

def main():
    data_sc, data_routes, data_changes = load_data()
    data_sc, le = preprocess_data(data_sc, data_routes)
    features, labels = create_features_labels(data_sc)
    model = train_and_predict(features, labels)

if __name__ == "__main__":
    main()
