import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "dataset/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path)

# URL-based features
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

# -------------------------------------------------
# 1. Class Distribution
# -------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="label"
)

plt.title("Distribution of Phishing and Legitimate URLs")
plt.xlabel("Label (0 = Phishing, 1 = Legitimate)")
plt.ylabel("Number of URLs")

plt.tight_layout()
plt.show()


# -------------------------------------------------
# 2. URL Length Distribution
# -------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="URLLength",
    hue="label",
    bins=50,
    kde=True
)

plt.title("URL Length Distribution")
plt.xlabel("URL Length")
plt.ylabel("Number of URLs")

plt.tight_layout()
plt.show()


# -------------------------------------------------
# 3. Number of Subdomains
# -------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="label",
    y="NoOfSubDomain"
)

plt.title("Number of Subdomains by Class")
plt.xlabel("Label (0 = Phishing, 1 = Legitimate)")
plt.ylabel("Number of Subdomains")

plt.tight_layout()
plt.show()


# -------------------------------------------------
# 4. HTTPS Usage
# -------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="IsHTTPS",
    hue="label"
)

plt.title("HTTPS Usage by URL Class")
plt.xlabel("HTTPS (0 = No, 1 = Yes)")
plt.ylabel("Number of URLs")

plt.tight_layout()
plt.show()