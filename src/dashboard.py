import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("models/random_forest.pkl")
encoders = joblib.load("models/encoders.pkl")

st.set_page_config(
    page_title="AdGrass",
    page_icon="📢",
    layout="wide"
)

st.title("AdGrass")
st.subheader("Real-Time Ad Recommendation Engine")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 25)

    gender = st.selectbox(
        "Gender",
        ["M", "F"]
    )

    sport = st.selectbox(
        "Favourite Sport",
        ["Cricket", "Football"]
    )

with col2:
    watch_time = st.slider(
        "Watch Time",
        0,
        180,
        60
    )

    previous_clicks = st.slider(
        "Previous Ad Clicks",
        0,
        20,
        3
    )

    device = st.selectbox(
        "Device",
        ["Mobile", "Laptop", "TV"]
    )

if st.button("Generate Recommendation"):

    user = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Sport": sport,
        "WatchTime": watch_time,
        "PreviousClicks": previous_clicks,
        "Device": device
    }])

    for col in ["Gender", "Sport", "Device"]:
        user[col] = encoders[col].transform(user[col])

    user = user[
        [
            "Age",
            "Gender",
            "Sport",
            "WatchTime",
            "PreviousClicks",
            "Device"
        ]
    ]

    prediction = model.predict(user)
    probabilities = model.predict_proba(user)

    ad = encoders["AdCategory"].inverse_transform(
        prediction
    )[0]

    confidence = max(probabilities[0])

    st.success(f"Recommended Ad: {ad}")
    ads = {
    "Gaming": {
        "title": "Gaming Laptop Sale",
        "description": "Up to 40% off gaming laptops."
    },
    "Sports": {
        "title": "Nike Running Shoes",
        "description": "Performance footwear for athletes."
    },
    "Food": {
        "title": "Food Delivery Offer",
        "description": "50% off on your first order."
    },
    "Travel": {
        "title": "Flight Booking Deal",
        "description": "Special discounts on flights."
    }
}
    st.subheader(ads[ad]["title"])
    st.write(ads[ad]["description"])

    st.metric(
        "Confidence",
        f"{confidence:.2%}"
    )

    st.subheader("Ad Ranking")

    for ad_name, prob in zip(
        encoders["AdCategory"].classes_,
        probabilities[0]
    ):
        st.progress(float(prob))
        st.write(
            f"{ad_name}: {prob:.2%}"
        )