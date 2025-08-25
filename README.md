# Asset Tracking ML Project

A Python project for **tracking assets and detecting anomalies** using machine learning.  
This project generates synthetic asset tracking data, trains both **unsupervised** (Isolation Forest) and **supervised** (Random Forest) models, and predicts anomalies.

---

## 🗂 Project Structure

asset-traking/
├── main.py # Entry point
├── data_generator.py # Generates synthetic asset tracking data
├── models.py # Defines ML models
├── train.py # Trains Isolation Forest & Random Forest
├── predict.py # Predicts anomalies on data
├── preprocessing.py # Prepares features & scaling
└── asset_data.csv # Generated CSV (not pushed if in .gitignore)
