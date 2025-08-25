import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(file="asset_data.csv"):
    df = pd.read_csv(file, parse_dates=["timestamp"])

    # Feature engineering
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.dayofweek

    features = ["latitude", "longitude", "speed", "hour", "day"]
    X = df[features]
    y = df["anomaly"]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

if __name__ == "__main__":
    X, y, scaler = load_and_preprocess()
    print(X[:5], y[:5])
