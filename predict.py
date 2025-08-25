import pandas as pd
from preprocessing import load_and_preprocess
from models import get_classifier

def predict_new_data(new_file="asset_data.csv"):
    X, y, scaler = load_and_preprocess(new_file)
    clf = get_classifier()
    clf.fit(X, y)  # for demo; normally load trained model

    preds = clf.predict(X)
    df = pd.read_csv(new_file)
    df["predicted_anomaly"] = preds
    print(df.head(10))
    return df

if __name__ == "__main__":
    predict_new_data()
