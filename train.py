from preprocessing import load_and_preprocess
from models import get_anomaly_model, get_classifier
from sklearn.metrics import classification_report

def train_models():
    X, y, _ = load_and_preprocess()

    # Unsupervised
    iso = get_anomaly_model()
    iso.fit(X)
    preds = iso.predict(X)
    preds = [1 if p == -1 else 0 for p in preds]  # map -1 → anomaly

    print("Isolation Forest Report:")
    print(classification_report(y, preds))

    # Supervised
    clf = get_classifier()
    clf.fit(X, y)
    preds2 = clf.predict(X)

    print("Random Forest Report:")
    print(classification_report(y, preds2))

    return iso, clf

if __name__ == "__main__":
    train_models()
