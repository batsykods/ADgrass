import pandas as pd
import random

rows = []

sports = ["Cricket", "Football"]
devices = ["Mobile", "Laptop", "TV"]
genders = ["M", "F"]

for _ in range(5000):

    age = random.randint(18, 60)
    gender = random.choice(genders)
    sport = random.choice(sports)
    watch_time = random.randint(5, 180)
    previous_clicks = random.randint(0, 20)
    device = random.choice(devices)

    if age < 25 and previous_clicks > 8:
        ad = "Gaming"
    elif sport == "Cricket" and age > 30:
        ad = "Food"
    elif watch_time > 100:
        ad = "Sports"
    else:
        ad = "Travel"

    rows.append([
        age,
        gender,
        sport,
        watch_time,
        previous_clicks,
        device,
        ad
    ])

df = pd.DataFrame(
    rows,
    columns=[
        "Age",
        "Gender",
        "Sport",
        "WatchTime",
        "PreviousClicks",
        "Device",
        "AdCategory"
    ]
)

df.to_csv("data/ad_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())