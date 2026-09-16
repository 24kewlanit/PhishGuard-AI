import joblib

print("========================================")
print("       MODEL INSPECTION")
print("========================================")

# Load trained model
model = joblib.load("phishing_model.pkl")

print("\nModel type:")
print(type(model))

print("\nNumber of features expected:")

if hasattr(model, "n_features_in_"):
    print(model.n_features_in_)
else:
    print("Not available")

print("\nFeature names stored in model:")

if hasattr(model, "feature_names_in_"):

    for feature in model.feature_names_in_:
        print(" -", feature)

else:

    print("No feature names stored in model")


print("\n========================================")
print("       FEATURES.PKL INSPECTION")
print("========================================")

# Load feature list
features = joblib.load("features.pkl")

print("\nFeature list type:")
print(type(features))

print("\nFeatures used by your model:")

for feature in features:
    print(" -", feature)

print("\nTotal features:", len(features))

print("\n========================================")