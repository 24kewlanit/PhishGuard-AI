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

# Show legitimate URLs with short URLs
legitimate = df[df["label"] == 1]

legitimate = legitimate.sort_values("URLLength")

print("\nLEGITIMATE URL EXAMPLES")
print("=" * 80)

print(
    legitimate[["URL"] + features]
    .head(15)
    .to_string(index=False)
)