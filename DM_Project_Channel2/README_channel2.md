# Channel II — PM2.5 Air Quality Prediction

Part of: **Predictive Modeling of Wildfire Activity and PM2.5 Air Quality Trends**  
Sadia Islam · Sowjanya Tammali · Missouri S&T · Spring 2026

---

## What This Channel Does

Predicts next-month county-level PM2.5 air quality in California (2015–2020) using EPA AQS data.  
Introduces **WWTR** — a custom XGBoost model with asymmetric loss that penalises PM2.5 underestimation more than overestimation, reducing missed pollution spike warnings.  
Connects to Channel I via a lag-correlation linking analysis.

---

## Folder Structure

```
DM_project/
├── data/
│   ├── raw/                        ← 6 zip files: daily_88101_2015–2020.zip
│   └── processed/                  ← auto-created by scripts
├── results/                        ← all outputs (CSV + PNG)
├── channel1_scripts/               ← cloned from Channel I repo
│   └── data/processed/
│       └── california_county_fire_final_ml_dataset.csv
├── pm25_step1_load.py
├── pm25_step2_aggregate.py
├── pm25_step3_features.py
├── pm25_step4_models.py
├── pm25_step5_linking.py
├── channel2_final.py               ← evaluation + all plots
├── wwtr_comparison.py              ← WWTR vs baseline comparison
└── requirements.txt
```

---

## Setup

```bash
cd DM_project/
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Apple Silicon (M1/M2/M3):** replace `tensorflow` with `tensorflow-macos tensorflow-metal`.

---

## Run Order

```bash
python pm25_step1_load.py          # load + filter California
python pm25_step2_aggregate.py     # county-month aggregation
python pm25_step3_features.py      # feature engineering + targets
python pm25_step4_models.py        # RF, XGBoost, LSTM + spike metrics
python pm25_step5_linking.py       # merge with Channel I + correlation
python channel2_final.py           # evaluation plots (6 figures)
python wwtr_comparison.py          # WWTR vs baseline head-to-head
```

---

## Scripts — What Each Does

### `pm25_step1_load.py`
Loads the 6 EPA AQS zip files (2015–2020), filters to `State Name == "California"`, parses dates.  
**Output:** `california_pm25_daily_2015_2020.csv`

### `pm25_step2_aggregate.py`
Groups daily station readings by `(NAME, year, month)`.  
Computes `avg_pm25`, `max_pm25`, `std_pm25`, `days_above_35` (days exceeding EPA 35.4 µg/m³).  
**Output:** `california_pm25_county_monthly.csv`

### `pm25_step3_features.py`
Builds autoregressive and seasonal features:

| Feature | What it captures |
|---|---|
| `pm25_lag1`, `pm25_lag2` | Previous 1–2 months' PM2.5 |
| `pm25_rolling3` | 3-month trailing average |
| `days_above_lag1` | Previous month's bad-air days |
| `month_sin`, `month_cos` | Seasonal cycle encoding |

Defines two targets:
- `next_month_pm25` — regression target
- `label_next_month_pm25` — binary (1 if next month ≥ P75)

**Output:** `california_pm25_final_ml_dataset.csv`

### `pm25_step4_models.py`
Trains three baseline models (train 2015–2018, test 2019–2020):
- **Random Forest** — 200 trees, max_depth 15
- **XGBoost (MSE)** — standard symmetric loss, 500 estimators
- **LSTM** — two-layer (64→32), 3-month sequences, early stopping

Spike threshold is set to **P75 of test-set PM2.5** (not 35.4 — monthly averages rarely exceed the daily EPA limit). Threshold saved to `results/spike_threshold.csv` so all scripts use the same value.

Computes for every model: MAE, RMSE, R², Spike Precision, Spike Recall, Spike F1, Missed Warnings.  
**Output:** `results/pm25_model_results.csv`, `results/pm25_predictions_2019_2020.csv`

### `pm25_step5_linking.py`
Merges Channel II output with Channel I (`california_county_fire_final_ml_dataset.csv`) on `(NAME, year, month)`.

Runs three analyses:
1. **Lag sweep** — Pearson r between `fire_count`/`avg_frp` at lag 0–3 months and `avg_pm25`. Identifies peak causal window.
2. **Partial correlation** — removes seasonal effect (`month_sin/cos`) from both variables before correlating. Controls for summer confound.
3. **Linking models** — Ridge Regression, Gradient Boosting, VAR + Granger causality. VAR tests whether fire_count Granger-causes PM2.5 directionally.

Key finding from paper: `avg_frp` at lag 0 has r = 0.39 (p < 0.001); all lags significant.  
**Output:** `results/lag_sweep_correlation.csv`, `results/partial_correlation.csv`, `results/linking_model_comparison.csv`

### `channel2_final.py`
Main evaluation script. Trains RF, XGBoost, WWTR and produces 6 publication-quality plots:

| Plot | What it shows |
|---|---|
| 1. Actual vs Predicted scatter | Overall prediction quality across all models |
| 2. Underestimation bias on spike months | Mean `y_true − y_pred` where PM2.5 > threshold — WWTR's direct target |
| 3. Quantile loss (q=0.9) | Asymmetric penalty favoring underestimation detection |
| 4. Residual box plot (spike months) | Full error distribution on dangerous months |
| 5. Time-series forecast | California monthly mean — WWTR tracks peaks more closely |
| 6. Directional accuracy + spike recall | Trend direction correctness vs dangerous-month detection rate |

**Output:** `results/channel2_evaluation.png`

### `wwtr_comparison.py`
Head-to-head: WWTR vs baseline XGBoost.  
WWTR predictions are clipped to `[y_train.min(), y_train.max()]` to correct for custom-loss output drift.  
Prints underestimation bias on spike months as a backup metric if threshold-based recall is close.  
**Output:** `results/wwtr_vs_xgb_comparison.csv`, `results/wwtr_vs_xgb_comparison.png`

---

## WWTR — Custom Model Summary

Standard XGBoost minimises symmetric MSE. WWTR changes two things:

**Sample weight** — upweights fire-season months:
```
w_i = 1 + β × (days_above_35_i / max(days_above_35))    β = 2.0
```

**Asymmetric loss** — underestimation costs 2.5× more:
```
L = w_i × [α × max(0, y − ŷ)²  +  max(0, ŷ − y)²]      α = 2.5
```

Gradient and Hessian are implemented via `xgb.train(obj=wwtr_loss)`.  
Result from paper: WWTR achieves highest spike recall among Channel II models.

---

## Key Results (from paper)

**Channel II model comparison:**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 4.41 | 9.96 | 0.070 |
| XGBoost (MSE) | 4.23 | 9.86 | 0.088 |
| LSTM | 4.67 | 10.24 | 0.018 |
| **WWTR** | 4.48 | 9.96 | 0.070 |

WWTR achieves highest spike recall — captures the most PM2.5 spike events.

**Linking analysis:**

| Fire variable | Lag | Pearson r | Significant |
|---|---|---|---|
| avg_frp | 0 | 0.390 | Yes |
| avg_frp | 1 | 0.189 | Yes |
| fire_count | 2 | 0.090 | Yes |

VAR/Granger model: MAE = 3.325, R² = 0.451 — temporal dependence matters.

---

## Merge Key (Channel I ↔ Channel II)

```python
merged = pd.merge(pm25_df, fire_df, on=["NAME", "year", "month"])
```

Channel I file needed: `channel1_scripts/data/processed/california_county_fire_final_ml_dataset.csv`

---

## Output Files Reference

| File | Description |
|---|---|
| `california_pm25_final_ml_dataset.csv` | ML-ready county-month dataset |
| `pm25_model_results.csv` | All model metrics incl. spike recall |
| `spike_threshold.csv` | P75 threshold used across all scripts |
| `pm25_predictions_2019_2020.csv` | XGBoost predictions on test set |
| `wildfire_pm25_merged.csv` | Combined Channel I + II dataset |
| `lag_sweep_correlation.csv` | Pearson r at lag 0–3 months |
| `partial_correlation.csv` | Seasonality-controlled correlations |
| `linking_model_comparison.csv` | Ridge / GB / VAR metrics |
| `channel2_evaluation.png` | 6-panel evaluation figure |
| `wwtr_vs_xgb_comparison.png` | WWTR vs baseline 3-panel figure |
