import pandas as pd

df = pd.read_csv("data/ad_data.csv")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nAd Category Distribution:")
print(df["AdCategory"].value_counts())
