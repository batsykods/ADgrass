import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/ad_data.csv")

encoders = {}

for col in ["Gender", "Sport", "Device", "AdCategory"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("AdCategory", axis=1)
print("Training columns:")
print(X.columns.tolist())

y = df["AdCategory"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(classification_report(y_test, predictions))

joblib.dump(model, "models/random_forest.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("Model saved successfully.")