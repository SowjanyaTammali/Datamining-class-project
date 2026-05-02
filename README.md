# Wildfire Prediction Pipeline — Channel I

## Project Goal

This part of the project focuses on **county-level wildfire activity prediction** for California using NASA FIRMS VIIRS satellite fire detection data.

The main research question is:

> Can we predict whether a California county will experience high wildfire activity in the next month using historical wildfire patterns?

This wildfire prediction output will later be used by the PM2.5 team to study how wildfire activity affects air quality.

---

## Study Setup

- **Region:** California, USA
- **Years:** 2015–2020
- **Spatial level:** County
- **Temporal level:** Monthly
- **Task type:** Binary classification

The target label is:

```text
1 = High wildfire activity next month
0 = Not high wildfire activity next month
```

High wildfire activity is defined using the **75th percentile threshold** of monthly fire count.

---

## Input Data

### 1. NASA FIRMS VIIRS-SNPP Active Fire Data

Raw input files:

```text
viirs-snpp_2015_United_States.csv
viirs-snpp_2016_United_States.csv
viirs-snpp_2017_United_States.csv
viirs-snpp_2018_United_States.csv
viirs-snpp_2019_United_States.csv
viirs-snpp_2020_United_States.csv
```

Important columns used:

```text
latitude
longitude
acq_date
frp
```

Each row represents one satellite-detected active fire point.

### 2. U.S. Census County Shapefile

Used to map each fire point to a California county.

```text
tl_2020_us_county/
```

---

## Overall Pipeline

```text
Raw FIRMS CSV files
        ↓
Filter California fire points
        ↓
Map fire points to counties
        ↓
Aggregate by county-month
        ↓
Create next-month wildfire label
        ↓
Create temporal and seasonal features
        ↓
Train ML models
        ↓
Generate wildfire risk prediction results
```

---

## Step-by-Step Processing

### Step 1: Combine FIRMS Files and Filter California

**Script:**

```text
wildfire_step1.py
```

**Input:**

```text
viirs-snpp_2015_United_States.csv
viirs-snpp_2016_United_States.csv
viirs-snpp_2017_United_States.csv
viirs-snpp_2018_United_States.csv
viirs-snpp_2019_United_States.csv
viirs-snpp_2020_United_States.csv
```

**Output:**

```text
california_fire_points_2015_2020.csv
california_fire_monthly_2015_2020.csv
```

**What this script does:**

- Loads all VIIRS-SNPP fire detection files from 2015–2020
- Combines them into one dataset
- Filters only California fire detections using latitude and longitude
- Saves California-only fire points
- Creates a basic monthly California-wide fire summary

---

### Step 2: Assign Fire Points to Counties

**Script:**

```text
wildfire_county.py
```

**Input:**

```text
california_fire_points_2015_2020.csv
tl_2020_us_county/
```

**Output:**

```text
california_county_fire_monthly_2015_2020.csv
```

**Output columns:**

```text
NAME
year
month
fire_count
avg_frp
```

**What this script does:**

- Converts wildfire latitude/longitude points into geospatial points
- Loads the U.S. county shapefile
- Filters California counties
- Uses spatial join to assign each fire point to a county
- Aggregates wildfire activity by county, year, and month

Column meanings:

- `NAME`: county name
- `fire_count`: number of fire detections in that county-month
- `avg_frp`: average Fire Radiative Power, used as a fire intensity feature

---

### Step 3: Create Prediction Label

**Script:**

```text
wildfire_label.py
```

**Input:**

```text
california_county_fire_monthly_2015_2020.csv
```

**Output:**

```text
california_county_fire_prediction_dataset.csv
```

**What this script does:**

- Creates `next_month_fire`
- Creates `label_next_month`
- Uses the 75th percentile fire-count threshold to define high wildfire activity

Label definition:

```text
label_next_month = 1 if next_month_fire >= 46
label_next_month = 0 otherwise
```

The threshold used was:

```text
46 fire detections
```

Label distribution:

```text
0: 2569
1: 880
```

At first, we tried using:

```text
1 = any fire next month
0 = no fire next month
```

However, almost every county-month had some fire detection, so nearly all labels became 1. To make the task meaningful, we changed the target to **high wildfire activity** instead of **any wildfire activity**.

---

### Step 4: Feature Engineering

**Script:**

```text
wildfire_features.py
```

**Input:**

```text
california_county_fire_prediction_dataset.csv
```

**Output:**

```text
california_county_fire_final_ml_dataset.csv
```

Features created:

```text
fire_count
avg_frp
fire_last_1
fire_last_2
frp_last_1
month_sin
month_cos
fire_rolling_3
```

Target column:

```text
label_next_month
```

Feature meanings:

- `fire_count`: current month fire detections
- `avg_frp`: current month average fire intensity
- `fire_last_1`: fire count from previous month
- `fire_last_2`: fire count from two months ago
- `frp_last_1`: previous month fire intensity
- `month_sin`, `month_cos`: seasonal month encoding
- `fire_rolling_3`: rolling average of fire count over 3 months

This is the final dataset used for machine learning.

---

## Final ML Dataset

Main output file:

```text
california_county_fire_final_ml_dataset.csv
```

Important columns:

```text
NAME
year
month
fire_count
avg_frp
next_month_fire
label_next_month
fire_last_1
fire_last_2
frp_last_1
month_sin
month_cos
fire_rolling_3
```

This is the most important file for the PM2.5 team.

The PM2.5 team can merge their dataset using:

```text
NAME, year, month
```

---

## Machine Learning Setup

Task:

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

This is a time-based split to avoid leakage from future data.

---

## Models Trained and Results

### 1. Random Forest

**Script:**

```text
wildfire_model.py
```

**Result:**

```text
Accuracy: 0.78
Class 1 Precision: 0.72
Class 1 Recall: 0.40
Class 1 F1-score: 0.52
```

**Interpretation:**

Random Forest achieved good overall accuracy, but it missed many high wildfire months. Its recall for high wildfire activity was low.

---

### 2. Balanced Random Forest

**Script:**

```text
wildfire_model_balanced.py
```

**Result:**

```text
Accuracy: 0.78
Class 1 Precision: 0.74
Class 1 Recall: 0.38
Class 1 F1-score: 0.50
```

**Interpretation:**

Adding class balancing did not improve high wildfire detection. The recall became slightly worse.

---

### 3. XGBoost

**Script:**

```text
wildfire_model_xgb.py
```

**Result:**

```text
Accuracy: 0.76
Class 1 Precision: 0.59
Class 1 Recall: 0.57
Class 1 F1-score: 0.58
```

**Interpretation:**

XGBoost gave the best full-test-set balance between accuracy and high wildfire activity detection. It improved recall compared with Random Forest.

---

### 4. Custom XGBoost with Threshold Tuning

**Script:**

```text
scripts/wildfire_model_xgb_custom.py
```

**Result:**

```text
Best threshold: 0.35
Accuracy: 0.71
Class 1 Precision: 0.51
Class 1 Recall: 0.70
Class 1 F1-score: 0.59
```

**Interpretation:**

This custom XGBoost model uses threshold tuning to optimize for the best F1-score on the positive class. It achieves higher recall (0.70) than the standard XGBoost (0.57) at the cost of lower precision (0.51 vs 0.59), resulting in similar F1-score. The tuned threshold of 0.35 prioritizes detecting more high wildfire months.

---

### 4. LSTM

**Scripts:**

```text
wildfire_lstm_dataset.py
wildfire_lstm_model.py
```

Input sequence shape:

```text
X shape: (3162, 3, 8)
y shape: (3162,)
```

This means:

```text
3162 samples
3 months per sequence
8 features per month
```

Improved LSTM result after normalization and class weighting:

```text
Accuracy: 0.65
Class 1 Precision: 0.36
Class 1 Recall: 0.64
Class 1 F1-score: 0.46
```

**Interpretation:**

LSTM detected more high wildfire months than XGBoost because it had higher recall. However, it also produced more false positives, so its precision and F1-score were lower.

---

## KumoRFM Exploratory Experiment

KumoRFM was tested as a relational foundation model using the Kumo Playground UI.

This was exploratory because full SDK/backend access was not available.

### Kumo Tables

County table:

```text
kumo_county_table.csv
```

Columns:

```text
county_id
NAME
```

Fire event table:

```text
kumo_fire_event_rfm_hidden.csv
```

Important columns:

```text
row_id
county_id
year
month
event_date
fire_count
avg_frp
fire_last_1
fire_last_2
frp_last_1
month_sin
month_cos
fire_rolling_3
label_next_month
```

### Kumo Playground Table Settings

For `kumo_county_table`:

```text
Primary Key: county_id
Time Column: None
```

For `kumo_fire_event_rfm_hidden`:

```text
Primary Key: row_id
Time Column: event_date
```

Relationship:

```text
kumo_fire_event_rfm_hidden.county_id → kumo_county_table.county_id
```

### Kumo Predictive Query

```text
PREDICT kumo_fire_event_rfm_hidden.label_next_month
FOR EACH kumo_fire_event_rfm_hidden.row_id
```

### KumoRFM Evaluation

Due to Playground export limitations, only a 100-row subset of 2020 predictions was exported and evaluated.

Subset result:

```text
Evaluated rows: 100
Accuracy: 1.00
Class 1 Precision: 1.00
Class 1 Recall: 1.00
Class 1 F1-score: 1.00
```

Important note:

This KumoRFM result is preliminary. It should not be interpreted as full test-set performance because it was evaluated only on a 100-row subset, not all 2020 rows.

---

## Results Summary

| Model | Evaluation Scope | Accuracy | Class 1 Recall | Class 1 F1-score |
|---|---:|---:|---:|---:|
| Random Forest | Full test set | 0.78 | 0.40 | 0.52 |
| Balanced Random Forest | Full test set | 0.78 | 0.38 | 0.50 |
| XGBoost | Full test set | 0.76 | 0.57 | 0.58 |
| Custom XGBoost | Full test set | 0.71 | 0.70 | 0.59 |
| LSTM | Full test set | 0.65 | 0.64 | 0.46 |
| KumoRFM | 100-row subset | 1.00 | 1.00 | 1.00 |

---

## Main Findings

1. The custom XGBoost with threshold tuning achieves the highest recall (0.70) for detecting high wildfire months among fully evaluated models.
2. XGBoost is the strongest fully evaluated model so far in terms of F1-score balance.
3. Random Forest gives good accuracy but misses many high wildfire months.
4. LSTM improves recall for high wildfire months but gives more false alarms.
5. KumoRFM shows promising relational prediction behavior, but full evaluation requires complete batch export or SDK backend access.

---

## Channel II — PM2.5 Modeling

The PM2.5 modeling channel will be handled separately.

Expected input:

```text
EPA AQS PM2.5 data
```

Expected final PM2.5 output format:

```text
NAME, year, month, avg_pm25
```

or:

```text
county, year, month, pm25_mean
```

The wildfire and PM2.5 channels can later be merged using:

```text
NAME, year, month
```

The combined dataset can then be used to study:

> Does wildfire activity help predict PM2.5 concentration at the county-month level?

---

## Output for PM2.5 Team

The main output file for the PM2.5 team is:

```text
california_county_fire_final_ml_dataset.csv
```

Recommended wildfire columns for PM2.5 modeling:

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

Recommended merge keys:

```text
NAME, year, month
```

---

## End Goal

The end goal of this wildfire channel is to produce county-month wildfire risk signals that can be used for air quality and health impact modeling.

Future extensions:

- Merge wildfire features with PM2.5 data
- Predict PM2.5 increases from wildfire activity
- Add asthma emergency visit data
- Study wildfire → PM2.5 → health impact
- Add climate features such as temperature, wind, and drought indicators
- Add graph-based county adjacency modeling

---

## Important Files

### Main Scripts

```text
wildfire_step1.py
wildfire_county.py
wildfire_label.py
wildfire_features.py
wildfire_model.py
wildfire_model_balanced.py
wildfire_model_xgb.py
wildfire_lstm_dataset.py
wildfire_lstm_model.py
prepare_kumo_rfm_files.py
prepare_kumo_eval_files.py
evaluate_kumo_subset.py
```

### Main Output Files

```text
california_county_fire_monthly_2015_2020.csv
california_county_fire_prediction_dataset.csv
california_county_fire_final_ml_dataset.csv
kumo_county_table.csv
kumo_fire_event_rfm_hidden.csv
kumo_actual_2020_labels.csv
kumo_predictions_2020_part1.csv
```

### Raw Files

```text
viirs-snpp_2015_United_States.csv
viirs-snpp_2016_United_States.csv
viirs-snpp_2017_United_States.csv
viirs-snpp_2018_United_States.csv
viirs-snpp_2019_United_States.csv
viirs-snpp_2020_United_States.csv
tl_2020_us_county/
```

---

## How to Run

Run scripts in this order:

```bash
python3 wildfire_step1.py
python3 wildfire_county.py
python3 wildfire_label.py
python3 wildfire_features.py
python3 wildfire_model.py
python3 wildfire_model_balanced.py
python3 wildfire_model_xgb.py
python3 wildfire_lstm_dataset.py
python3 wildfire_lstm_model.py
python3 prepare_kumo_rfm_files.py
python3 prepare_kumo_eval_files.py
python3 evaluate_kumo_subset.py
```

---

## Suggested GitHub Structure

```text
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── kumo/
│
├── scripts/
│
├── results/
│
├── README.md
└── .gitignore
```

---

## GitHub Notes

Do not push virtual environments or unnecessary system files.

Recommended `.gitignore`:

```text
kumoenv/
path/
__pycache__/
.DS_Store
*.npy
.env
```

Large raw CSV files may also be excluded if GitHub file size becomes a problem.

---

## Current Status

Completed:

- NASA FIRMS data loading
- California filtering
- County-level spatial join
- Monthly aggregation
- Binary label creation
- Feature engineering
- Random Forest model
- Balanced Random Forest model
- XGBoost model
- LSTM model
- KumoRFM exploratory test

Current best fully evaluated model:

```text
XGBoost
```
