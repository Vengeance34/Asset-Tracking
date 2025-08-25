from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier

def get_anomaly_model():
    """Unsupervised anomaly detection"""
    return IsolationForest(contamination=0.1, random_state=42)

def get_classifier():
    """Supervised anomaly classification"""
    return RandomForestClassifier(n_estimators=100, random_state=42)
