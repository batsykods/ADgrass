import joblib

model = joblib.load("models/random_forest.pkl")

print(model.feature_names_in_)