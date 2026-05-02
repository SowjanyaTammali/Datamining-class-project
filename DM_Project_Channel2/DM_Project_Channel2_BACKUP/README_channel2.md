# Channel II — PM2.5 Air Quality Prediction
## Setup & Run Guide (Mac + VS Code)

---

## Project Folder Structure

```
project/
├── data/
│   ├── raw/                          ← place the 6 zip files here
│   └── processed/                    ← auto-created by scripts
├── results/                          ← auto-created by scripts
├── channel1_scripts/                 ← your existing Channel I scripts
├── pm25_step1_load.py
├── pm25_step2_aggregate.py
├── pm25_step3_features.py
├── pm25_step4_models.py
├── pm25_step5_linking.py
└── requirements_channel2.txt
```

---

## One-Time Setup (Mac Terminal or VS Code Terminal)

### 1. Create virtual environment
```bash
cd project/
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements_channel2.txt
```

> TensorFlow on Apple Silicon Mac:
> If you have M1/M2/M3, replace `tensorflow` with:
> ```bash
> pip install tensorflow-macos tensorflow-metal
> ```

### 3. Configure VS Code Python Interpreter
- Open VS Code → `Cmd+Shift+P` → "Python: Select Interpreter"
- Choose the `venv` interpreter (path: `./venv/bin/python`)

### 4. Place the zip files
Copy the 6 zip files into `data/raw/`:
```
data/raw/daily_88101_2015.zip
data/raw/daily_88101_2016.zip
...
data/raw/daily_88101_2020.zip
```
(The naming prefix like `1777742016484_` is fine — the script auto-detects via glob)

---

## Run Order

Run each script from VS Code terminal (with venv active):

```bash
# Step 1: Load and filter California data (~1-2 min)
python pm25_step1_load.py

# Step 2: Aggregate to county-month
python pm25_step2_aggregate.py

# Step 3: Feature engineering
python pm25_step3_features.py

# Step 4: Train RF, XGBoost, LSTM models
python pm25_step4_models.py

# Step 5: Link with Channel I (run AFTER Channel I is complete)
python pm25_step5_linking.py
```

---

## Important: Step 5 Prerequisites

Step 5 requires Channel I's output file to exist at:
```
data/processed/california_county_fire_final_ml_dataset.csv
```
Run all Channel I scripts first, then run Step 5.

---

## Output Files

| File | Description |
|------|-------------|
| `data/processed/california_pm25_daily_2015_2020.csv` | California daily PM2.5 |
| `data/processed/california_pm25_county_monthly.csv` | County-month aggregation |
| `data/processed/california_pm25_final_ml_dataset.csv` | Final ML-ready dataset |
| `results/pm25_model_results.csv` | RF / XGBoost / LSTM metrics |
| `results/pm25_predictions_2019_2020.csv` | XGBoost predictions on test set |
| `results/wildfire_pm25_merged.csv` | Merged Channel I + II dataset |
| `results/lag_correlation_results.csv` | Wildfire-PM2.5 correlation table |
| `results/feature_importance_combined.csv` | SHAP-style importance |
| `results/linking_analysis.png` | Visualization for paper |

---

## Merge Keys (Channel I ↔ Channel II)

Both channels use: **NAME, year, month**
- `NAME` = California county name (e.g., "Los Angeles")
- `year` = 2015–2020
- `month` = 1–12

---

## Evaluation Metrics

**Regression (PM2.5 prediction):**
- MAE — Mean Absolute Error (µg/m³)
- RMSE — Root Mean Squared Error
- R² — Variance explained

**Classification (high PM2.5 label):**
- Accuracy, Precision, Recall, F1-score (Class 1)

---

## Research Question Answered by Step 5

> "Does wildfire activity (fire_count, avg_frp, lag features) significantly
>  correlate with and predict next-month PM2.5 at the county level?"

The `lag_correlation_results.csv` and `feature_importance_combined.csv`
provide the quantitative evidence for your paper's findings section.
