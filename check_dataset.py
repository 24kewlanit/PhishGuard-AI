import pandas as pd

file_path = "dataset/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")

print("\nLabel values:")
print(df["label"].unique())

print("\nNumber of URLs in each class:")
print(df["label"].value_counts())

print("\nPercentage of each class:")
print(df["label"].value_counts(normalize=True) * 100)