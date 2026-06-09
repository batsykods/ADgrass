from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("models/random_forest.pkl")
encoders = joblib.load("models/encoders.pkl")


@app.route("/")
def home():
    return {"message": "AdGrass API Running"}


@app.route("/recommend", methods=["POST"])
def recommend():

    user = request.json

    df = pd.DataFrame([user])

    for col in ["Gender", "Sport", "Device"]:
        df[col] = encoders[col].transform(df[col])

    prediction = model.predict(df)
    probabilities = model.predict_proba(df)

    recommended_ad = encoders["AdCategory"].inverse_transform(
        prediction
    )[0]

    confidence = float(max(probabilities[0]))

    return jsonify({
        "recommended_ad": recommended_ad,
        "confidence": round(confidence, 4)
    })


if __name__ == "__main__":
    app.run(debug=True)