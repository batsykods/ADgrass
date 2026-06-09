import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/random_forest.pkl")

df = pd.read_csv("data/ad_data.csv")

features = [
    "Age",
    "Gender",
    "Sport",
    "WatchTime",
    "PreviousClicks",
    "Device"
]

importance = model.feature_importances_

plt.figure(figsize=(8,5))
plt.bar(features, importance)

plt.title("AdGrass Feature Importance")
plt.ylabel("Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()