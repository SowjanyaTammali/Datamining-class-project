import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("california_county_fire_final_ml_dataset.csv")

features = [
    "fire_count",
    "avg_frp",
    "fire_last_1",
    "fire_last_2",
    "frp_last_1",
    "month_sin",
    "month_cos",
    "fire_rolling_3"
]

train_df = df[df["year"] <= 2018]
test_df  = df[df["year"] > 2018]

X_train = train_df[features]
y_train = train_df["label_next_month"]

X_test = test_df[features]
y_test = test_df["label_next_month"]

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=3,   # helps minority class
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))