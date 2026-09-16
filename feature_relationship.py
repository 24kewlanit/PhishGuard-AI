import pandas as pd

# Load dataset
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

# Calculate correlation with target
correlation = df[features + ["label"]].corr()["label"]

# Remove label itself
correlation = correlation.drop("label")

# Sort by absolute correlation
correlation = correlation.reindex(
    correlation.abs().sort_values(ascending=False).index
)

print("\nFEATURE CORRELATION WITH LABEL")
print("=" * 50)

print(correlation)