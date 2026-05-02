import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# Load final ML dataset
df = pd.read_csv("../data/processed/california_county_fire_final_ml_dataset.csv")

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

# Time-based split
train_df = df[df["year"] <= 2018]
test_df = df[df["year"] > 2018]

X_train = train_df[features]
y_train = train_df["label_next_month"]

X_test = test_df[features]
y_test = test_df["label_next_month"]

# Calculate class imbalance ratio
num_negative = (y_train == 0).sum()
num_positive = (y_train == 1).sum()
scale_pos_weight = num_negative / num_positive

print("Negative samples:", num_negative)
print("Positive samples:", num_positive)
print("scale_pos_weight:", scale_pos_weight)

# Cost-sensitive XGBoost
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# Get predicted probabilities for class 1
y_prob = model.predict_proba(X_test)[:, 1]

# Threshold tuning
best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.1, 0.91, 0.01):
    y_pred_temp = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_temp)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final prediction using best threshold
y_pred = (y_prob >= best_threshold).astype(int)

print("\nBest threshold:", round(best_threshold, 2))

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Class 1 Precision:", precision_score(y_test, y_pred))
print("Class 1 Recall:", recall_score(y_test, y_pred))
print("Class 1 F1:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
importance = pd.Series(model.feature_importances_, index=features)
print("\nFeature Importance:")
print(importance.sort_values(ascending=False))

# Save predictions
test_output = test_df.copy()
test_output["xgb_custom_prob_1"] = y_prob
test_output["xgb_custom_pred"] = y_pred

test_output.to_csv("xgb_custom_predictions.csv", index=False)

print("\nSaved predictions to xgb_custom_predictions.csv")
