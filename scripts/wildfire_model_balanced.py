import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("california_county_fire_final_ml_dataset.csv")

features = [
    "fire_count",
    "avg_frp",
    "fire_last_1",
    "fire_last_2",
    "frp_last_1"
]

# Time-based split
train_df = df[df["year"] <= 2018]
test_df  = df[df["year"] > 2018]

X_train = train_df[features]
y_train = train_df["label_next_month"]

X_test = test_df[features]
y_test = test_df["label_next_month"]

# ---- BALANCED RANDOM FOREST ----
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Feature importance
importance = pd.Series(model.feature_importances_, index=features)
print("\nFeature Importance:\n")
print(importance.sort_values(ascending=False))