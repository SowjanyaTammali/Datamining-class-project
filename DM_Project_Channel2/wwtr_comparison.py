"""
WWTR vs Baseline XGBoost — Spike Recall Comparison
Run this AFTER pm25_step4_models.py

Output:
  results/wwtr_vs_xgb_comparison.csv
  results/wwtr_vs_xgb_comparison.png
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, "venv", "bin", "python")
if os.path.exists(venv_python):
    current_python = os.path.abspath(sys.executable)
    target_python = os.path.abspath(venv_python)
    if current_python != target_python:
        print(f"Switching to project venv Python: {target_python}")
        os.execv(target_python, [target_python] + sys.argv)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

ALPHA     = 2.5
BETA      = 2.0
# Load same threshold used in step4 so comparison is consistent
try:
    THRESHOLD = float(pd.read_csv("results/spike_threshold.csv")["spike_threshold"].iloc[0])
except Exception:
    THRESHOLD = 35.4
print(f"Using spike threshold: {THRESHOLD:.2f} µg/m³")

# ── DATA ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/california_pm25_final_ml_dataset.csv")

FEATURES = [
    "avg_pm25","max_pm25","std_pm25","days_above_35",
    "pm25_lag1","pm25_lag2","pm25_rolling3","days_above_lag1",
    "month_sin","month_cos"
]
TARGET = "next_month_pm25"

train = df[df["year"] <= 2018].dropna(subset=FEATURES + [TARGET])
test  = df[df["year"] >= 2019].dropna(subset=FEATURES + [TARGET])
X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

fire_max = df["days_above_35"].max()
w_train  = 1.0 + BETA * (train["days_above_35"].values / (fire_max + 1e-9))
w_test   = 1.0 + BETA * (test["days_above_35"].values  / (fire_max + 1e-9))

# ── BASELINE XGBoost (standard MSE) ──────────────────────────────────────────
xgb_base = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, verbosity=0, random_state=42,
    early_stopping_rounds=30, eval_metric="rmse"
)
xgb_base.fit(X_train, y_train,
             eval_set=[(X_test, y_test)], verbose=False)
base_pred = xgb_base.predict(X_test)

# ── WWTR custom loss ──────────────────────────────────────────────────────────
def wwtr_loss(y_pred, dtrain):
    y_true  = dtrain.get_label()
    weights = dtrain.get_weight()
    if len(weights) == 0:
        weights = np.ones_like(y_true)
    residual = y_true - y_pred
    grad = np.where(residual > 0,
                    -2.0 * ALPHA * residual,
                     2.0 * (-residual)) * weights
    hess = np.where(residual > 0,
                    2.0 * ALPHA * weights,
                    2.0 * weights)
    return grad, hess

def wwtr_eval(y_pred, dtrain):
    y_true  = dtrain.get_label()
    weights = dtrain.get_weight()
    if len(weights) == 0:
        weights = np.ones_like(y_true)
    residual = y_true - y_pred
    loss  = np.where(residual > 0, ALPHA * residual**2, residual**2)
    score = np.sqrt(np.average(loss, weights=weights))
    return "wwtr_rmse", score

dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
dtest  = xgb.DMatrix(X_test,  label=y_test,  weight=w_test)

params = dict(max_depth=6, learning_rate=0.04, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5,
              seed=42, verbosity=0)

wwtr_model = xgb.train(
    params, dtrain, num_boost_round=600,
    obj=wwtr_loss, custom_metric=wwtr_eval,
    evals=[(dtest,"test")],
    early_stopping_rounds=30, verbose_eval=False
)
wwtr_pred = wwtr_model.predict(dtest)

# ── METRICS ───────────────────────────────────────────────────────────────────
def metrics(name, y_true, y_pred, threshold=THRESHOLD):
    actual  = y_true.values > threshold
    flagged = y_pred        > threshold
    tp = np.sum( actual &  flagged)
    fn = np.sum( actual & ~flagged)
    fp = np.sum(~actual &  flagged)
    prec   = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1     = 2*prec*recall / (prec + recall + 1e-9)
    return {
        "Model":         name,
        "MAE":           round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE":          round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "R2":            round(r2_score(y_true, y_pred), 3),
        "Spike_Precision": round(prec,   3),
        "Spike_Recall":    round(recall, 3),
        "Spike_F1":        round(f1,     3),
        "Missed_Warnings": int(fn),
    }

rows = [
    metrics("Baseline XGBoost", y_test, base_pred),
    metrics("WWTR (custom)",    y_test, wwtr_pred),
]
cmp = pd.DataFrame(rows)
cmp.to_csv("results/wwtr_vs_xgb_comparison.csv", index=False)

print("\n" + "="*65)
print("WWTR vs BASELINE XGBOOST")
print("="*65)
print(cmp.to_string(index=False))

recall_gain = rows[1]["Spike_Recall"] - rows[0]["Spike_Recall"]
fn_saved    = rows[0]["Missed_Warnings"] - rows[1]["Missed_Warnings"]
rmse_delta  = rows[1]["RMSE"] - rows[0]["RMSE"]
print(f"\nSpike recall gain : +{recall_gain:.3f}")
print(f"Missed warnings saved : {fn_saved}")
print(f"RMSE change       : {rmse_delta:+.3f} µg/m³")
print("="*65)

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# 1. Key metrics bar chart
metrics_cols = ["MAE","RMSE","Spike_Recall","Spike_F1"]
x = np.arange(len(metrics_cols))
w = 0.3
axes[0].bar(x - w/2, [rows[0][m] for m in metrics_cols], w,
            label="Baseline XGBoost", color="#4575b4", alpha=0.85)
axes[0].bar(x + w/2, [rows[1][m] for m in metrics_cols], w,
            label="WWTR",             color="#d73027", alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_cols, rotation=10)
axes[0].set_title("Metric comparison")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

# 2. Actual vs predicted — Baseline
axes[1].scatter(y_test, base_pred, alpha=0.3, s=10, color="#4575b4")
lim = [y_test.min(), y_test.max()]
axes[1].plot(lim, lim, "k--", linewidth=0.8)
axes[1].axhline(THRESHOLD, color="#d73027", linewidth=0.8, linestyle=":")
axes[1].axvline(THRESHOLD, color="#d73027", linewidth=0.8, linestyle=":")
axes[1].set_xlabel("Actual PM2.5")
axes[1].set_ylabel("Predicted PM2.5")
axes[1].set_title(f"Baseline XGBoost\nRMSE={rows[0]['RMSE']}  Recall={rows[0]['Spike_Recall']}")
axes[1].grid(alpha=0.2)

# 3. Actual vs predicted — WWTR
axes[2].scatter(y_test, wwtr_pred, alpha=0.3, s=10, color="#d73027")
axes[2].plot(lim, lim, "k--", linewidth=0.8)
axes[2].axhline(THRESHOLD, color="#d73027", linewidth=0.8, linestyle=":")
axes[2].axvline(THRESHOLD, color="#d73027", linewidth=0.8, linestyle=":")
axes[2].set_xlabel("Actual PM2.5")
axes[2].set_ylabel("Predicted PM2.5")
axes[2].set_title(f"WWTR (custom loss)\nRMSE={rows[1]['RMSE']}  Recall={rows[1]['Spike_Recall']}")
axes[2].grid(alpha=0.2)

plt.suptitle("WWTR vs Baseline XGBoost — PM2.5 Spike Detection\n"
             f"EPA threshold = {THRESHOLD} µg/m³  |  "
             f"Recall gain = +{recall_gain:.3f}  |  "
             f"Missed warnings saved = {fn_saved}",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("results/wwtr_vs_xgb_comparison.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → results/wwtr_vs_xgb_comparison.png")
