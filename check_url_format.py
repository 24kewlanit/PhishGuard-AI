import pandas as pd

file_path = "dataset/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path)

print("\nFIRST 10 URL VALUES")
print("=" * 60)

for i in range(10):
    print(repr(df.loc[i, "URL"]))

print("\nURL LENGTHS")
print("=" * 60)

for i in range(10):
    url = df.loc[i, "URL"]
    print("URL:", repr(url))
    print("Length:", len(url))
    print()