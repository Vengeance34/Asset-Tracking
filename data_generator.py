import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_asset_data(num_assets=5, num_records=500):
    """Generate synthetic asset tracking data."""
    data = []
    base_time = datetime.now()

    for asset_id in range(1, num_assets + 1):
        lat, lon = 37.7749, -122.4194  # Start near San Francisco
        for i in range(num_records):
            timestamp = base_time + timedelta(minutes=i)
            speed = abs(np.random.normal(40, 10))  # avg speed 40 ± 10
            lat += np.random.normal(0.001, 0.002)  # small drift
            lon += np.random.normal(0.001, 0.002)

            # Introduce anomalies (10% chance)
            anomaly = 1 if random.random() < 0.1 else 0
            if anomaly:
                lat += np.random.uniform(0.5, 1.0)  # sudden jump
                lon += np.random.uniform(0.5, 1.0)
                speed *= random.uniform(2, 4)

            data.append([f"Asset_{asset_id}", timestamp, lat, lon, speed, anomaly])

    df = pd.DataFrame(data, columns=["asset_id", "timestamp", "latitude", "longitude", "speed", "anomaly"])
    return df

if __name__ == "__main__":
    df = generate_asset_data()
    print(df.head())
    df.to_csv("asset_data.csv", index=False)
