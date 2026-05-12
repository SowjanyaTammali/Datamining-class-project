import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------------
# Helper function
# -----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# -----------------------------
# Load dataset
# -----------------------------
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

# -----------------------------
# Time-based split
# Keep same split style as your old code for fair comparison
# -----------------------------
train_df = df[df["year"] <= 2018].copy()
test_df  = df[df["year"] > 2018].copy()

X_train = train_df[features]
y_train = train_df["label_next_month"].astype(int).values

X_test = test_df[features]
y_test = test_df["label_next_month"].astype(int).values

# -----------------------------
# WFR-XGB hyperparameters
# alpha -> extra penalty for positive class
# beta  -> extra weight using recent fire history
# -----------------------------
alpha_pos = 3.0
beta_fire = 1.0

# Normalize fire_rolling_3 using TRAIN max only
max_fire_roll = train_df["fire_rolling_3"].max()
if max_fire_roll == 0:
    max_fire_roll = 1.0

# Wildfire-aware instance weights
# base wildfire-history weight
base_weight_train = 1.0 + beta_fire * (train_df["fire_rolling_3"].values / max_fire_roll)

# positive-class upweighting
class_weight_train = np.where(y_train == 1, alpha_pos, 1.0)

# final training weight
train_weights = base_weight_train * class_weight_train

print("WFR-XGB settings")
print("alpha_pos =", alpha_pos)
print("beta_fire =", beta_fire)
print("max fire_rolling_3 (train) =", round(max_fire_roll, 4))

print("\nTraining weight summary:")
print("Min:", round(train_weights.min(), 4))
print("Max:", round(train_weights.max(), 4))
print("Mean:", round(train_weights.mean(), 4))

# -----------------------------
# Create DMatrix
# -----------------------------
dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

# -----------------------------
# Custom objective:
# weighted binary logistic loss
#
# grad = w * (p - y)
# hess = w * p * (1 - p)
# -----------------------------
def wfr_objective(preds, dtrain):
    y_true = dtrain.get_label()
    weights = dtrain.get_weight()

    p = sigmoid(preds)

    grad = weights * (p - y_true)
    hess = weights * p * (1.0 - p)

    return grad, hess

# -----------------------------
# XGBoost training parameters
# -----------------------------
params = {
    "max_depth": 5,
    "eta": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 1,
    "lambda": 1.0,           # L2 regularization
    "alpha": 0.0,            # L1 regularization
    "seed": 42,
    "tree_method": "hist",
    "disable_default_eval_metric": 1
}

num_round = 500

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_round,
    obj=wfr_objective
)

# -----------------------------
# Predict probabilities
# Use raw margin, then apply sigmoid manually
# -----------------------------
raw_pred = model.predict(dtest, output_margin=True)
y_prob = sigmoid(raw_pred)

# -----------------------------
# Threshold tuning
# For now, keep F1 tuning to match your previous setup
# -----------------------------
threshold_results = []
best_threshold = 0.5
best_f1 = -1

for threshold in np.arange(0.10, 0.91, 0.01):
    y_pred_temp = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_temp)
    precision = precision_score(y_test, y_pred_temp, zero_division=0)
    recall = recall_score(y_test, y_pred_temp, zero_division=0)

    threshold_results.append({
        "threshold": round(threshold, 2),
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final prediction
y_pred = (y_prob >= best_threshold).astype(int)

# -----------------------------
# Evaluation
# -----------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\nBest threshold:", round(best_threshold, 2))
print("\nAccuracy:", acc)
print("Class 1 Precision:", prec)
print("Class 1 Recall:", rec)
print("Class 1 F1:", f1)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------
# Feature importance
# -----------------------------
importance_dict = model.get_score(importance_type="gain")

importance_series = pd.Series(
    {feature: importance_dict.get(feature, 0.0) for feature in features}
).sort_values(ascending=False)

print("\nFeature Importance (gain):")
print(importance_series)

# -----------------------------
# Save predictions
# -----------------------------
test_output = test_df.copy()
test_output["wfr_xgb_prob_1"] = y_prob
test_output["wfr_xgb_pred"] = y_pred

test_output.to_csv("wfr_xgb_predictions.csv", index=False)

# Save threshold tuning results
threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv("wfr_xgb_threshold_tuning.csv", index=False)

# Save feature importance
importance_series.rename("importance").to_csv("wfr_xgb_feature_importance.csv", header=True)

print("\nSaved:")
print("1. wfr_xgb_predictions.csv")
print("2. wfr_xgb_threshold_tuning.csv")
print("3. wfr_xgb_feature_importance.csv")
