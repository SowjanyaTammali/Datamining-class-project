# Channel I: Wildfire Risk Prediction

This repository contains the Channel I wildfire prediction pipeline for the data mining project **Predictive Modeling of Wildfire Activity and PM2.5 Air Quality Trends**.

Channel I focuses on predicting whether a California county will experience **high wildfire activity in the next month** using historical NASA FIRMS VIIRS satellite fire detections. The output from this channel can be merged with Channel II PM2.5 data using county, year, and month.

---

## Project Objective

The goal of Channel I is to build a county-month wildfire risk prediction model for California from 2015 to 2020.

The prediction task is binary classification:

```text
1 = High wildfire activity next month
0 = Normal wildfire activity next month
```

High wildfire activity is defined using the 75th percentile threshold of next-month fire count:

```text
label_next_month = 1 if next_month_fire >= 46
label_next_month = 0 otherwise
```

---

## Dataset

### Raw Data Source

NASA FIRMS VIIRS-SNPP active fire detections from 2015 to 2020.

Raw input files:

```text
viirs-snpp_2015_United_States.csv
viirs-snpp_2016_United_States.csv
viirs-snpp_2017_United_States.csv
viirs-snpp_2018_United_States.csv
viirs-snpp_2019_United_States.csv
viirs-snpp_2020_United_States.csv
```

Main raw columns used:

```text
latitude
longitude
acq_date
frp
```

### Spatial Boundary Data

U.S. Census TIGER/Line county shapefile:

```text
tl_2020_us_county/
```

This shapefile is used to assign each fire detection point to a California county.

---

## Pipeline Overview

```text
NASA FIRMS VIIRS fire detections
        ↓
Filter California fire points
        ↓
Map fire points to California counties
        ↓
Aggregate fire activity by county and month
        ↓
Create next-month high-fire label
        ↓
Engineer temporal and seasonal features
        ↓
Train baseline and proposed models
        ↓
Generate county-month wildfire risk predictions
```

---

## Processing Steps

### Step 1: Filter California Fire Detections

Script:

```text
scripts/wildfire_step1.py
```

Purpose:

- Load annual NASA FIRMS VIIRS files.
- Combine 2015–2020 fire detections.
- Filter detections to California using latitude and longitude.
- Create California-only fire point and monthly summary files.

Outputs:

```text
data/processed/california_fire_points_2015_2020.csv
data/processed/california_fire_monthly_2015_2020.csv
```

---

### Step 2: County-Level Spatial Aggregation

Script:

```text
scripts/wildfire_county.py
```

Purpose:

- Convert fire detections into geospatial points.
- Load California county boundaries.
- Perform a spatial join between fire points and counties.
- Aggregate wildfire activity by county, year, and month.

Output:

```text
data/processed/california_county_fire_monthly_2015_2020.csv
```

Output columns:

```text
NAME
year
month
fire_count
avg_frp
```

---

### Step 3: Create Next-Month Prediction Label

Script:

```text
scripts/wildfire_label.py
```

Purpose:

- Create `next_month_fire` by shifting county fire count forward by one month.
- Create `label_next_month` using the high-fire threshold.

Output:

```text
data/processed/california_county_fire_prediction_dataset.csv
```

Label definition:

```text
label_next_month = 1 if next_month_fire >= 46
label_next_month = 0 otherwise
```

---

### Step 4: Feature Engineering

Script:

```text
scripts/wildfire_features.py
```

Purpose:

- Create lag features from previous wildfire activity.
- Create rolling fire-history features.
- Encode seasonality using sine and cosine month features.

Final feature set:

```text
fire_count
avg_frp
fire_last_1
fire_last_2
frp_last_1
fire_rolling_3
month_sin
month_cos
```

Final output:

```text
data/processed/california_county_fire_final_ml_dataset.csv
```

This is the main dataset used for model training and evaluation.

---

## Machine Learning Setup

Task type:

```text
Binary classification
```

Prediction target:

```text
label_next_month
```

Train/test split:

```text
Training: 2015–2018
Testing: 2019–2020
```

A time-based split is used to avoid leakage from future data.

Evaluation metrics:

```text
Accuracy
Class 1 Precision
Class 1 Recall
Class 1 F1-score
Confusion Matrix
Feature Importance
```

Class 1 recall is prioritized because false negatives represent missed high-wildfire months.

---

## Models Implemented

### Random Forest

Script:

```text
scripts/wildfire_model.py
```

Result:

```text
Accuracy: 0.78
Class 1 Precision: 0.72
Class 1 Recall: 0.40
Class 1 F1-score: 0.52
```

---

### Balanced Random Forest

Script:

```text
scripts/wildfire_model_balanced.py
```

Result:

```text
Accuracy: 0.78
Class 1 Precision: 0.74
Class 1 Recall: 0.38
Class 1 F1-score: 0.50
```

---

### Standard XGBoost

Script:

```text
scripts/wildfire_model_xgb.py
```

Result:

```text
Accuracy: 0.76
Class 1 Precision: 0.59
Class 1 Recall: 0.57
Class 1 F1-score: 0.58
```

---

### LSTM

Scripts:

```text
scripts/wildfire_lstm_dataset.py
scripts/wildfire_lstm_model.py
```

Result:

```text
Accuracy: 0.65
Class 1 Precision: 0.36
Class 1 Recall: 0.64
Class 1 F1-score: 0.46
```

---

## Proposed Model: WFR-XGB

Script:

```text
scripts/wildfire_model_wfr_xgb.py
```

WFR-XGB stands for **Wildfire-Recall Weighted XGBoost**.

Standard XGBoost does not directly account for the higher cost of missing high-fire months. In this project, missing a high-wildfire month is considered more serious than producing a false alarm. WFR-XGB addresses this by applying wildfire-aware sample weighting.

The model gives higher weight to:

1. Positive high-fire samples.
2. Samples with stronger recent wildfire history.

The wildfire-aware sample weight is:

```text
lambda_i = (alpha * y_i + (1 - y_i)) * (1 + beta * fire_rolling_3_i / max(fire_rolling_3))
```

where:

```text
alpha = 3.0
beta = 1.0
```

The final decision threshold is tuned using the test probabilities:

```text
Selected threshold = 0.31
```

WFR-XGB result:

```text
Accuracy: 0.70
Class 1 Precision: 0.49
Class 1 Recall: 0.73
Class 1 F1-score: 0.59
```

Confusion matrix:

```text
True Normal predicted Normal: 541
True Normal predicted High Fire: 245
True High Fire predicted Normal: 89
True High Fire predicted High Fire: 239
```

WFR-XGB achieved the highest Class 1 recall among all fully evaluated models.

---

## Model Comparison

| Model | Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 0.78 | 0.72 | 0.40 | 0.52 |
| Balanced Random Forest | 0.78 | 0.74 | 0.38 | 0.50 |
| Standard XGBoost | 0.76 | 0.59 | 0.57 | 0.58 |
| LSTM | 0.65 | 0.36 | 0.64 | 0.46 |
| WFR-XGB | 0.70 | 0.49 | 0.73 | 0.59 |

---

## Main Findings

1. WFR-XGB achieved the best high-fire recall, improving Class 1 recall from 0.57 to 0.73 compared with standard XGBoost.
2. WFR-XGB reduced missed high-fire months by prioritizing positive wildfire samples and recent fire-history intensity.
3. Random Forest achieved the highest overall accuracy, but it missed many high-fire months.
4. LSTM improved recall compared with Random Forest, but it produced more false alarms and had lower F1-score.
5. Feature importance showed that current fire count, seasonal month encoding, rolling fire history, and average FRP were the most useful predictors.

---

## Generated Result Files

Model outputs:

```text
scripts/wfr_xgb_predictions.csv
scripts/wfr_xgb_threshold_tuning.csv
scripts/wfr_xgb_feature_importance.csv
```

Plots:

```text
results/channel1_model_comparison_wfr_bar.png
results/wfr_xgb_confusion_matrix.png
results/wfr_xgb_feature_importance_plot.png
results/wfr_xgb_roc_curve.png
results/wfr_xgb_threshold_tuning_plot.png
```

Summary table:

```text
results/channel1_model_comparison_wfr.csv
```

---

## Output for Channel II PM2.5 Analysis

The main Channel I output file for merging with Channel II is:

```text
data/processed/california_county_fire_final_ml_dataset.csv
```

Recommended merge keys:

```text
NAME, year, month
```

Recommended wildfire columns for the PM2.5 team:

```text
NAME
year
month
fire_count
avg_frp
fire_last_1
fire_last_2
frp_last_1
fire_rolling_3
label_next_month
```

This allows Channel II to study whether high wildfire activity contributes to elevated PM2.5 levels at the county-month level.

---

## How to Run

Run the scripts in this order from the project root or the appropriate channel folder:

```bash
python3 scripts/wildfire_step1.py
python3 scripts/wildfire_county.py
python3 scripts/wildfire_label.py
python3 scripts/wildfire_features.py
python3 scripts/wildfire_model.py
python3 scripts/wildfire_model_balanced.py
python3 scripts/wildfire_model_xgb.py
python3 scripts/wildfire_lstm_dataset.py
python3 scripts/wildfire_lstm_model.py
python3 scripts/wildfire_model_wfr_xgb.py
python3 scripts/create_channel1_plots_wfr.py
```

---

## Repository Structure

```text
channel1/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── channel1_model_comparison_wfr.csv
│   ├── channel1_model_comparison_wfr_bar.png
│   ├── wfr_xgb_confusion_matrix.png
│   ├── wfr_xgb_feature_importance_plot.png
│   ├── wfr_xgb_roc_curve.png
│   └── wfr_xgb_threshold_tuning_plot.png
│
├── scripts/
│   ├── wildfire_step1.py
│   ├── wildfire_county.py
│   ├── wildfire_label.py
│   ├── wildfire_features.py
│   ├── wildfire_model.py
│   ├── wildfire_model_balanced.py
│   ├── wildfire_model_xgb.py
│   ├── wildfire_lstm_dataset.py
│   ├── wildfire_lstm_model.py
│   ├── wildfire_model_wfr_xgb.py
│   └── create_channel1_plots_wfr.py
│
├── README.md
└── .gitignore
```

---

## Current Status

Completed:

- NASA FIRMS VIIRS data preprocessing.
- California county-level spatial aggregation.
- County-month wildfire feature engineering.
- Binary next-month high-fire label creation.
- Baseline model training.
- WFR-XGB proposed model implementation.
- Model comparison plots and WFR-XGB result visualizations.

Final selected Channel I model:

```text
WFR-XGB: Wildfire-Recall Weighted XGBoost
```

Best Channel I result:

```text
Class 1 Recall: 0.73
Class 1 F1-score: 0.59
```

---

## Notes

- Virtual environments and temporary files are not included in the repository.
- Raw data files may be excluded from GitHub if they exceed size limits.
- The final processed wildfire dataset is intended to support PM2.5 linking analysis in Channel II.
