"""
Channel II - Step 5: LINKING ANALYSIS — Wildfire → PM2.5 Correlation
Implements Option 2 (Lag-Correlation Bridge) from the research design.

Input:
  - results/pm25_predictions_2019_2020.csv       (Channel II output)
  - data/processed/california_county_fire_final_ml_dataset.csv  (Channel I output)

Output:
  - results/wildfire_pm25_merged.csv
  - results/lag_correlation_results.csv
  - results/feature_importance_combined.csv

Research Question answered:
  At what spatial+temporal scale does wildfire activity most strongly
  predict PM2.5 concentration at the county-month level?
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD BOTH CHANNELS ───────────────────────────────────────────────────────
pm25_df = pd.read_csv("data/processed/california_pm25_final_ml_dataset.csv")
fire_df = pd.read_csv("data/processed/california_county_fire_final_ml_dataset.csv")

# ── MERGE ON (NAME, year, month) ─────────────────────────────────────────────
merged = pd.merge(
    pm25_df[["NAME","year","month","avg_pm25","next_month_pm25",
             "label_next_month_pm25","pm25_lag1","pm25_rolling3"]],
    fire_df[["NAME","year","month","fire_count","avg_frp",
             "fire_last_1","fire_last_2","frp_last_1",
             "fire_rolling_3","label_next_month"]],
    on=["NAME","year","month"],
    how="inner"
)
print(f"Merged dataset shape: {merged.shape}")
merged.to_csv(f"{OUTPUT_DIR}/wildfire_pm25_merged.csv", index=False)

# ── LAG-CORRELATION ANALYSIS ─────────────────────────────────────────────────
# Key question: does fire activity at lag 0, 1, 2 months correlate with PM2.5?
fire_cols = ["fire_count","avg_frp","fire_last_1","fire_last_2",
             "frp_last_1","fire_rolling_3"]
pm25_target = "avg_pm25"

lag_results = []
for fc in fire_cols:
    valid = merged[[fc, pm25_target]].dropna()
    r_p, p_p = pearsonr(valid[fc], valid[pm25_target])
    r_s, p_s = spearmanr(valid[fc], valid[pm25_target])
    lag_results.append({
        "fire_feature":  fc,
        "pearson_r":     round(r_p, 4),
        "pearson_p":     round(p_p, 4),
        "spearman_r":    round(r_s, 4),
        "spearman_p":    round(p_s, 4),
        "significant":   p_p < 0.05
    })

lag_df = pd.DataFrame(lag_results).sort_values("pearson_r", ascending=False)
print("\nLag-Correlation Results:")
print(lag_df.to_string(index=False))
lag_df.to_csv(f"{OUTPUT_DIR}/lag_correlation_results.csv", index=False)

# ── COMBINED FEATURE IMPORTANCE (XGBoost on merged data) ────────────────────
# This is the core finding: which wildfire features predict PM2.5?
FIRE_FEATS = ["fire_count","avg_frp","fire_last_1","fire_last_2",
              "frp_last_1","fire_rolling_3","label_next_month"]
PM25_FEATS = ["pm25_lag1","pm25_rolling3"]
ALL_FEATS  = FIRE_FEATS + PM25_FEATS

model_data = merged[ALL_FEATS + ["next_month_pm25","year"]].dropna()
train = model_data[model_data["year"] <= 2018]
test  = model_data[model_data["year"] >= 2019]

xgb_link = xgb.XGBRegressor(n_estimators=300, max_depth=5,
                              learning_rate=0.05, verbosity=0, random_state=42)
xgb_link.fit(train[ALL_FEATS], train["next_month_pm25"])
pred = xgb_link.predict(test[ALL_FEATS])

print(f"\nCombined Model (wildfire + PM2.5 features):")
print(f"  MAE : {mean_absolute_error(test['next_month_pm25'], pred):.3f}")
print(f"  R²  : {r2_score(test['next_month_pm25'], pred):.3f}")

importance_df = pd.DataFrame({
    "feature":    ALL_FEATS,
    "importance": xgb_link.feature_importances_
}).sort_values("importance", ascending=False)
print("\nFeature Importance (wildfire + PM2.5):")
print(importance_df.to_string(index=False))
importance_df.to_csv(f"{OUTPUT_DIR}/feature_importance_combined.csv", index=False)

# ── PLOT ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Correlation bar chart
colors = ["#d73027" if v > 0 else "#4575b4" for v in lag_df["pearson_r"]]
axes[0].barh(lag_df["fire_feature"], lag_df["pearson_r"], color=colors)
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].set_xlabel("Pearson r")
axes[0].set_title("Wildfire Feature Correlation with PM2.5")
axes[0].grid(axis="x", alpha=0.3)

# Feature importance bar chart
axes[1].barh(importance_df["feature"], importance_df["importance"], color="#2166ac")
axes[1].set_xlabel("Importance Score")
axes[1].set_title("Feature Importance: Combined Model\n(Wildfire + PM2.5 features → next month PM2.5)")
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/linking_analysis.png", dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {OUTPUT_DIR}/linking_analysis.png")
