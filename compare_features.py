import pandas as pd

df = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

features = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "NoOfSubDomain",
    "NoOfLettersInURL",
    "NoOfDegitsInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "IsHTTPS"
]

print("\nDATASET FEATURE VALUES")
print("=" * 70)

print(df[features + ["label"]].head(10).to_string(index=False))

print("\nFEATURE STATISTICS")
print("=" * 70)

print(df[features].describe().to_string())