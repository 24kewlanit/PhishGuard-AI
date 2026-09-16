import pandas as pd

file_path = "dataset/PhiUSIIL_Phishing_URL_Dataset.csv"

df = pd.read_csv(file_path)


def get_real_url(value):

    value = str(value)

    if "](" in value:
        value = value.split("](", 1)[1]

        if value.endswith(")"):
            value = value[:-1]

    return value


print("\nURL CONVERSION TEST")
print("=" * 60)

for i in range(5):

    original = df.loc[i, "URL"]

    real_url = get_real_url(original)

    print("\nOriginal:")
    print(original)

    print("Real URL:")
    print(real_url)

    print("Dataset URLLength:",
          df.loc[i, "URLLength"])

    print("Calculated URLLength:",
          len(real_url))