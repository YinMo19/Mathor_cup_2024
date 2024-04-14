import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

def load_data():
    data_sc = pd.read_csv('附件1.csv', encoding='gb2312')
    data_routes = pd.read_csv('附件3.csv', encoding='gb2312')
    data_changes = pd.read_csv('附件4.csv', encoding='gb2312')
    return data_sc, data_routes, data_changes

def preprocess_data(data_sc, data_routes, data_changes, existing_csc):
    data_sc['日期'] = pd.to_datetime(data_sc['日期'], format='%Y/%m/%d')
    le = LabelEncoder()
    data_sc['分拣中心'] = le.fit_transform(data_sc['分拣中心'])

    # Calculate the average departure and arrival for each sorting center
    avg_departure = data_routes.groupby('始发分拣中心')['货量'].mean().rename('平均发出货量').reset_index()
    avg_arrival = data_routes.groupby('到达分拣中心')['货量'].mean().rename('平均到达货量').reset_index()

    # Convert sorting center codes
    avg_departure['始发分拣中心'] = le.transform(avg_departure['始发分拣中心'])
    avg_arrival['到达分拣中心'] = le.transform(avg_arrival['到达分拣中心'])

    # Merge the adjusted volume information
    data_sc = pd.merge(data_sc, avg_departure, left_on='分拣中心', right_on='始发分拣中心', how='left')
    data_sc = pd.merge(data_sc, avg_arrival, left_on='分拣中心', right_on='到达分拣中心', how='left')
    data_sc.fillna({'平均发出货量': 0, '平均到达货量': 0}, inplace=True)

    return data_sc, le

def create_features_labels(data):
    data['year'] = data['日期'].dt.year
    data['month'] = data['日期'].dt.month
    data['day'] = data['日期'].dt.day
    data['weekday'] = data['日期'].dt.weekday
    features = data[['分拣中心', 'year', 'month', 'day', 'weekday', '平均发出货量', '平均到达货量']]
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

def predict_future(model, last_date, le, data_sc, existing_csc):
    # Generate future dates for the next 30 days
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    future_df = pd.DataFrame({
        'date': pd.to_datetime(list(future_dates) * len(existing_csc)),
        '分拣中心': existing_csc * len(future_dates)
    })
    
    future_df['year'] = future_df['date'].dt.year
    future_df['month'] = future_df['date'].dt.month
    future_df['day'] = future_df['date'].dt.day
    future_df['weekday'] = future_df['date'].dt.weekday

    # Estimate average departure and arrival volumes
    avg_departure = data_sc['平均发出货量'].mean()
    avg_arrival = data_sc['平均到达货量'].mean()

    future_df['平均发出货量'] = avg_departure
    future_df['平均到达货量'] = avg_arrival

    # Filter out sorting centers not in existing_csc
    future_df = future_df[future_df['分拣中心'].isin(existing_csc)]

    # Extract features
    features = future_df[['分拣中心', 'year', 'month', 'day', 'weekday', '平均发出货量', '平均到达货量']]
    
    # Use the model to make predictions
    predicted_values = model.predict(features)
    future_df['value'] = predicted_values
    future_df['date'] = future_df['date'].dt.strftime('%Y/%m/%d')  # Format the date

    results_df = future_df[['分拣中心', 'date', 'value']]
    results_df.rename(columns={'分拣中心': 'SC_ID'}, inplace=True)

    # Save to CSV
    results_df.to_csv('future_predictions.csv', index=False)
    print("未来预测结果已保存到 'future_predictions.csv'")

def main():
    data_sc, data_routes, data_changes = load_data()
    existing_csc = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37,
        38, 39, 40, 41, 43, 44, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56,
        57, 58, 60, 61, 63, 66, 68
    ]
    data_sc, le = preprocess_data(data_sc, data_routes, data_changes, existing_csc)
    features, labels = create_features_labels(data_sc)
    model = train_and_predict(features, labels)
    last_date = data_sc['日期'].max()  # Get the last known date
    predict_future(model, last_date, le, data_sc, existing_csc)

if __name__ == "__main__":
    main()
