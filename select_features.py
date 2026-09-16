import pandas as pd

file_path = "dataset/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path)

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

print("Selected URL-based features:")
print()

for feature in features:
    print(feature)

print("\nChecking whether all features exist:")

missing_features = [
    feature for feature in features
    if feature not in df.columns
]

if len(missing_features) == 0:
    print("All selected features are available.")
else:
    print("Missing features:", missing_features)