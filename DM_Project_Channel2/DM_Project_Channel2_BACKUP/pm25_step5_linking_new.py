"""
Channel II - Step 5: LINKING ANALYSIS — Wildfire → PM2.5 Correlation Bridge
Implements Option 2 (Lag-Correlation Bridge) — the published approach.

Core Research Question:
  At what temporal lag does wildfire intensity most strongly predict
  next-month PM2.5 concentration at the county level?

Novel Contributions:
  1. Explicit lag sweep (0–3 months) to identify peak causal lag
  2. Partial correlation controlling for seasonality (removes confounding)
  3. Uses Channel I model output (label_next_month) as a predictive signal
  4. Three linking models: Ridge, GradientBoosting, VAR (time-series)
  5. Publishable result: "FRP at lag-1 predicts PM2.5 with r=X, p<0.05"

Inputs:
  - data/processed/california_pm25_final_ml_dataset.csv
  - channel1_scripts/data/processed/california_county_fire_final_ml_dataset.csv

Outputs:
  - results/wildfire_pm25_merged.csv
  - results/lag_sweep_correlation.csv
  - results/partial_correlation.csv
  - results/linking_model_comparison.csv
  - results/feature_importance_combined.csv
  - results/linking_analysis.png
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & MERGE BOTH CHANNELS
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 5: WILDFIRE → PM2.5 LINKING ANALYSIS")
print("=" * 60)

pm25_df = pd.read_csv("data/processed/california_pm25_final_ml_dataset.csv")
fire_df = pd.read_csv(
    "channel1_scripts/data/processed/california_county_fire_final_ml_dataset.csv"
)

merged = pd.merge(
    pm25_df[["NAME","year","month","avg_pm25","next_month_pm25",
             "label_next_month_pm25","pm25_lag1","pm25_lag2",
             "pm25_rolling3","month_sin","month_cos"]],
    fire_df[["NAME","year","month","fire_count","avg_frp",
             "fire_last_1","fire_last_2","frp_last_1",
             "fire_rolling_3","label_next_month"]],
    on=["NAME","year","month"], how="inner"
)
print(f"\nMerged shape     : {merged.shape}")
print(f"Counties matched : {merged['NAME'].nunique()}")
print(f"Years covered    : {sorted(merged['year'].unique())}")
merged.to_csv(f"{OUTPUT_DIR}/wildfire_pm25_merged.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LAG SWEEP (0–3 months)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- LAG SWEEP ANALYSIS (0–3 months) ---")

df_lag = merged.sort_values(["NAME","year","month"]).copy()
grp = df_lag.groupby("NAME")

for lag in range(0, 4):
    df_lag[f"fire_count_lag{lag}"] = grp["fire_count"].shift(lag)
    df_lag[f"avg_frp_lag{lag}"]    = grp["avg_frp"].shift(lag)

lag_sweep = []
for lag in range(0, 4):
    for fire_var in ["fire_count","avg_frp"]:
        col = f"{fire_var}_lag{lag}"
        valid = df_lag[[col,"avg_pm25"]].dropna()
        if len(valid) < 30:
            continue
        r_p, p_p = pearsonr(valid[col], valid["avg_pm25"])
        r_s, p_s = spearmanr(valid[col], valid["avg_pm25"])
        lag_sweep.append({
            "fire_variable": fire_var, "lag_months": lag,
            "pearson_r": round(r_p,4), "pearson_p": round(p_p,6),
            "spearman_r": round(r_s,4),
            "significant": "YES" if p_p < 0.05 else "NO",
            "n_obs": len(valid)
        })

lag_df = pd.DataFrame(lag_sweep).sort_values(["fire_variable","lag_months"])
print(lag_df.to_string(index=False))
lag_df.to_csv(f"{OUTPUT_DIR}/lag_sweep_correlation.csv", index=False)

sig_lags = lag_df[lag_df["significant"] == "YES"]
if len(sig_lags) > 0:
    best = sig_lags.loc[sig_lags["pearson_r"].abs().idxmax()]
    best_lag = int(best["lag_months"])
    best_var = best["fire_variable"]
    print(f"\n★ Best lag: {best_var} at lag-{best_lag} months "
          f"(r={best['pearson_r']}, p={best['pearson_p']:.4f})")
else:
    best_lag, best_var = 1, "avg_frp"
    print("\nDefaulting to lag-1")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PARTIAL CORRELATION (controlling for seasonality)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n--- PARTIAL CORRELATION (controlling for month) ---")

def partial_correlation(df, x_col, y_col, control_cols):
    data = df[[x_col, y_col] + control_cols].dropna()
    def residuals(target, controls):
        X = data[controls].values
        y = data[target].values
        X = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return y - X @ beta
    r, p = pearsonr(residuals(x_col, control_cols), residuals(y_col, control_cols))
    return round(r,4), round(p,6)

controls = ["month_sin","month_cos"]
fire_features = ["fire_count","avg_frp","fire_last_1","frp_last_1","fire_rolling_3"]
partial_results = []
for fc in fire_features:
    valid = df_lag[[fc,"avg_pm25"] + controls].dropna()
    r_raw, p_raw   = pearsonr(valid[fc], valid["avg_pm25"])
    r_part, p_part = partial_correlation(df_lag, fc, "avg_pm25", controls)
    partial_results.append({
        "fire_feature": fc,
        "raw_pearson_r": round(r_raw,4), "raw_p": round(p_raw,6),
        "partial_r_adj_season": r_part, "partial_p": p_part,
        "significant": "YES" if p_part < 0.05 else "NO"
    })

partial_df = pd.DataFrame(partial_results)
print(partial_df.to_string(index=False))
partial_df.to_csv(f"{OUTPUT_DIR}/partial_correlation.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. THREE LINKING MODELS
# ═══════════════════════════════════════════════════════════════════════════════
FIRE_FEATS = [f"fire_count_lag{best_lag}", f"avg_frp_lag{best_lag}",
              "fire_last_1","fire_last_2","frp_last_1",
              "fire_rolling_3","label_next_month"]
PM25_FEATS = ["pm25_lag1","pm25_lag2","pm25_rolling3"]
SEASON     = ["month_sin","month_cos"]
ALL_FEATS  = FIRE_FEATS + PM25_FEATS + SEASON
TARGET     = "next_month_pm25"

model_data = df_lag[ALL_FEATS + [TARGET,"year","NAME"]].dropna()
train = model_data[model_data["year"] <= 2018]
test  = model_data[model_data["year"] >= 2019]
X_train, y_train = train[ALL_FEATS], train[TARGET]
X_test,  y_test  = test[ALL_FEATS],  test[TARGET]

print(f"\nTrain: {len(train):,} | Test: {len(test):,}")
linking_results = []

# A. Ridge Regression
print("\n--- Model A: Ridge Regression ---")
scaler  = StandardScaler()
ridge   = Ridge(alpha=1.0)
ridge.fit(scaler.fit_transform(X_train), y_train)
r_pred  = ridge.predict(scaler.transform(X_test))
r_mae   = mean_absolute_error(y_test, r_pred)
r_rmse  = np.sqrt(mean_squared_error(y_test, r_pred))
r_r2    = r2_score(y_test, r_pred)
print(f"MAE={r_mae:.3f}  RMSE={r_rmse:.3f}  R²={r_r2:.3f}")
coef_df = pd.DataFrame({"feature":ALL_FEATS,"coefficient":ridge.coef_}
                        ).sort_values("coefficient", key=abs, ascending=False)
print(coef_df.to_string(index=False))
linking_results.append({"Model":"Ridge Regression","MAE":r_mae,"RMSE":r_rmse,"R2":r_r2})

# B. Gradient Boosting
print("\n--- Model B: Gradient Boosting ---")
gb = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                learning_rate=0.05, subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
g_pred = gb.predict(X_test)
g_mae  = mean_absolute_error(y_test, g_pred)
g_rmse = np.sqrt(mean_squared_error(y_test, g_pred))
g_r2   = r2_score(y_test, g_pred)
print(f"MAE={g_mae:.3f}  RMSE={g_rmse:.3f}  R²={g_r2:.3f}")
linking_results.append({"Model":"Gradient Boosting","MAE":g_mae,"RMSE":g_rmse,"R2":g_r2})

fi_df = pd.DataFrame({"feature":ALL_FEATS,"importance":gb.feature_importances_}
                      ).sort_values("importance", ascending=False)
fi_df.to_csv(f"{OUTPUT_DIR}/feature_importance_combined.csv", index=False)

# C. VAR + Granger Causality
print("\n--- Model C: VAR (Vector Autoregression) ---")
try:
    ca_monthly = merged.groupby(["year","month"]).agg(
        avg_pm25   = ("avg_pm25",   "mean"),
        fire_count = ("fire_count", "mean")
    ).reset_index().sort_values(["year","month"])

    var_data  = ca_monthly[["avg_pm25","fire_count"]].values
    n_train   = 48
    var_train = var_data[:n_train]
    var_test  = var_data[n_train:]

    var_fit = VAR(var_train).fit(maxlags=3, ic="aic")
    print(f"VAR selected lag order: {var_fit.k_ar}")

    forecasts, history = [], list(var_train)
    for i in range(len(var_test)):
        fc = var_fit.forecast(np.array(history[-var_fit.k_ar:]), steps=1)
        forecasts.append(fc[0][0])
        history.append(var_test[i])

    v_actual = var_test[:,0]
    v_pred   = np.array(forecasts)
    v_mae    = mean_absolute_error(v_actual, v_pred)
    v_rmse   = np.sqrt(mean_squared_error(v_actual, v_pred))
    v_r2     = r2_score(v_actual, v_pred)
    print(f"MAE={v_mae:.3f}  RMSE={v_rmse:.3f}  R²={v_r2:.3f}")

    print("\nGranger Causality (fire_count → PM2.5):")
    gc = grangercausalitytests(ca_monthly[["avg_pm25","fire_count"]].values,
                               maxlag=3, verbose=False)
    for lag_n, res in gc.items():
        f_s, p_v = res[0]["ssr_ftest"][0], res[0]["ssr_ftest"][1]
        print(f"  Lag {lag_n}: F={f_s:.3f}, p={p_v:.4f} "
              f"→ {'✓ significant' if p_v < 0.05 else 'not significant'}")

    linking_results.append({"Model":"VAR","MAE":v_mae,"RMSE":v_rmse,"R2":v_r2})
except Exception as e:
    print(f"VAR error: {e} — run: pip install statsmodels")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SAVE & PLOT
# ═══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(linking_results)
print(f"\nLINKING MODEL COMPARISON\n{results_df.to_string(index=False)}")
results_df.to_csv(f"{OUTPUT_DIR}/linking_model_comparison.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Plot 1: Lag sweep
for var in ["fire_count","avg_frp"]:
    sub = lag_df[lag_df["fire_variable"] == var]
    axes[0,0].plot(sub["lag_months"], sub["pearson_r"], marker="o", label=var)
axes[0,0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0,0].set_xlabel("Lag (months)"); axes[0,0].set_ylabel("Pearson r")
axes[0,0].set_title("Lag Sweep: Wildfire → PM2.5 Correlation")
axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

# Plot 2: Raw vs partial correlation
x = np.arange(len(partial_df)); w = 0.35
axes[0,1].bar(x-w/2, partial_df["raw_pearson_r"],        w, label="Raw r",     color="#d73027")
axes[0,1].bar(x+w/2, partial_df["partial_r_adj_season"], w, label="Partial r", color="#4575b4")
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(partial_df["fire_feature"], rotation=20, ha="right")
axes[0,1].set_title("Raw vs Partial Correlation\n(seasonality controlled)")
axes[0,1].legend(); axes[0,1].grid(axis="y", alpha=0.3)

# Plot 3: Feature importance
axes[1,0].barh(fi_df["feature"][:10], fi_df["importance"][:10], color="#2166ac")
axes[1,0].set_title("Top Feature Importance\n(Gradient Boosting)")
axes[1,0].set_xlabel("Importance"); axes[1,0].grid(axis="x", alpha=0.3)

# Plot 4: Model comparison
xm = np.arange(len(results_df))
axes[1,1].bar(xm, results_df["MAE"], 0.4, label="MAE", color="#d73027", alpha=0.8)
ax2 = axes[1,1].twinx()
ax2.plot(xm, results_df["R2"], "o--", color="#1a9850", linewidth=2, label="R²")
axes[1,1].set_xticks(xm); axes[1,1].set_xticklabels(results_df["Model"], rotation=10)
axes[1,1].set_ylabel("MAE (µg/m³)", color="#d73027")
ax2.set_ylabel("R²", color="#1a9850")
axes[1,1].set_title("Linking Model Comparison")
axes[1,1].legend(loc="upper left"); ax2.legend(loc="upper right")
axes[1,1].grid(axis="y", alpha=0.3)

plt.suptitle("Wildfire → PM2.5 Linking Analysis | California 2015–2020",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/linking_analysis.png", dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {OUTPUT_DIR}/linking_analysis.png")

print("\n" + "=" * 60)
print("KEY FINDING FOR PAPER:")
if len(sig_lags) > 0:
    print(f"  {best_var} at lag-{best_lag} month(s) is the strongest")
    print(f"  predictor of PM2.5 (r={best['pearson_r']}, p={best['pearson_p']:.4f})")
print("  See: results/lag_sweep_correlation.csv")
print("  See: results/partial_correlation.csv")
print("=" * 60)
