import joblib
import pandas as pd

# Load model
model = joblib.load("models/random_forest.pkl")
encoders = joblib.load("models/encoders.pkl")

# Sample user
user = {
    "Age": 23,
    "Gender": "M",
    "Sport": "Football",
    "WatchTime": 150,
    "PreviousClicks": 12,
    "Device": "Mobile"
}

# Convert categorical values
input_data = pd.DataFrame([user])

for col in ["Gender", "Sport", "Device"]:
    input_data[col] = encoders[col].transform(input_data[col])

# Predict
probabilities = model.predict_proba(input_data)

ad_classes = encoders["AdCategory"].classes_

print("\nAd Ranking:")

for ad, prob in zip(ad_classes, probabilities[0]):
    print(f"{ad}: {prob:.2%}")