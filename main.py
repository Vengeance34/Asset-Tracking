from data_generator import generate_asset_data
from train import train_models
from predict import predict_new_data

def main():
    print("🚀 Generating data...")
    df = generate_asset_data()
    df.to_csv("asset_data.csv", index=False)
    print("✅ Data saved to asset_data.csv")

    print("🔧 Training models...")
    train_models()

    print("📡 Running predictions...")
    predict_new_data()

if __name__ == "__main__":
    main()
