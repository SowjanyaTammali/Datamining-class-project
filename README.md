# Predictive Modeling of Wildfire Activity and PM2.5 Air Quality Trends

<<<<<<< HEAD
This repository contains the code and project materials for a two-channel data mining project that studies wildfire activity and PM2.5 air-quality trends in California from 2015 to 2020.

## Project Overview

The project is organized into two complementary channels:

### Channel I: Wildfire Risk Prediction
Channel I predicts whether a California county will experience high wildfire activity in the next month using historical NASA FIRMS VIIRS fire detections.

Main tasks:
- Filter and process wildfire detections for California
- Aggregate wildfire activity at the county-month level
- Create next-month wildfire-risk labels
- Engineer temporal and seasonal wildfire features
- Train baseline and proposed wildfire prediction models

Folder:
- [`channel1/`](./channel1)

### Channel II: PM2.5 Air Quality Prediction
Channel II predicts future PM2.5 air-quality trends using EPA AQS PM2.5 observations and examines whether wildfire activity contributes to elevated PM2.5 levels.

Main tasks:
- Load and filter California PM2.5 data
- Aggregate PM2.5 observations at the county-month level
- Engineer lag, rolling, and seasonal PM2.5 features
- Train baseline and proposed PM2.5 prediction models
- Link wildfire and PM2.5 signals through correlation and time-series analysis

Folder:
- [`DM_Project_Channel2/`](./DM_Project_Channel2)

## How the Two Channels Connect

Both channels are designed at the same county-month level and can be merged using:
- `NAME`
- `year`
- `month`

This shared structure allows the project to:
- predict wildfire risk,
- predict PM2.5 air quality,
- and analyze whether wildfire intensity helps explain PM2.5 variation.
=======
This repository contains the work completed for the wildfire prediction and air-quality modeling project. The project is organized around two main components:

- Channel I: county-level wildfire activity prediction for California
- Channel II: PM2.5 linkage and downstream environmental analysis

A Kumo-based exploratory experiment was also included as part of the wildfire modeling workflow, and it is documented here as part of the project’s progress.

---

## Project Overview

### Channel I: Wildfire Prediction

The wildfire pipeline uses NASA FIRMS VIIRS active fire detections to predict whether a California county will experience high wildfire activity in the next month.

Key artifacts include:

- `channel1/data/processed/california_county_fire_final_ml_dataset.csv`
- `channel1/results/channel1_model_comparison_wfr.csv`
- `channel1/results/wfr_xgb_threshold_tuning.csv`

The primary modeling objective was to improve recall for high-fire months, since missed wildfire events are more costly than false alarms in this setting.

### KumoRFM Exploratory Experiment

A Kumo relational foundation model experiment was conducted to test whether a table-based relational model could predict the next-month wildfire label using county and event tables.

The Kumo workflow used the following data artifacts:

- `kumo_county_table.csv`
- `kumo_fire_event_rfm_hidden.csv`
- `kumo_actual_2020_labels.csv`
- `kumo_predictions_2020_part1.csv`

Schema used in the Kumo exploratory setup:

- `kumo_county_table`: county metadata keyed by `county_id`
- `kumo_fire_event_rfm_hidden`: county-month wildfire features and labels keyed by `row_id`

Relational linkage:

```text
kumo_fire_event_rfm_hidden.county_id -> kumo_county_table.county_id
```

Prediction query:

```text
PREDICT kumo_fire_event_rfm_hidden.label_next_month
FOR EACH kumo_fire_event_rfm_hidden.row_id
```

This experiment was exploratory and was evaluated on a 100-row subset from the 2020 period because full batch export or backend access was not available in the Playground workflow.

Preliminary KumoRFM result:

```text
Accuracy: 1.00
Class 1 Precision: 1.00
Class 1 Recall: 1.00
Class 1 F1-score: 1.00
```

This result is encouraging, but it should be interpreted as a promising pilot result rather than a full-model performance estimate.

### Channel II: PM2.5 Linking

The PM2.5 analysis uses the wildfire risk signals created in Channel I to assess whether wildfire activity is associated with elevated county-month PM2.5 conditions.

The primary merge key is:

```text
NAME, year, month
```

---

## Work Completed So Far

- NASA FIRMS VIIRS data preprocessing and California filtering
- County-level spatial aggregation by county and month
- Monthly wildfire feature creation and temporal lag engineering
- Binary next-month label construction for high wildfire activity
- Baseline ML modeling using Random Forest, Balanced Random Forest, XGBoost, and LSTM
- Proposed WFR-XGB wildfire-recall-weighted model
- KumoRFM exploratory relational modeling
- Result comparison and summary reporting

### Current best evaluated wildfire model

```text
WFR-XGB
```

Best reported recall:

```text
Class 1 Recall: 0.73
Class 1 F1-score: 0.59
```

Kumo remains valuable as an exploratory relational benchmark, but it would need full backend or complete batch export access before it could be treated as a final production result.

---

## Kumo Setup and Run Notes

The Kumo work in this project was designed as a relational-table exploratory test for wildfire forecasting. The workflow was built around preparing county and fire-event tables, then uploading them to the Kumo Playground or API for a prediction task.

### 1. Environment

Use the project virtual environment:

```bash
source kumoenv/bin/activate
```

If the Kumo SDK is needed, install the required package in the environment:

```bash
pip install kumoai
```

### 2. Prepare Kumo Data

From the relevant project folder, run the preparation scripts:

```bash
python3 DM_Project_Channel2/channel1_scripts/scripts/kumo_prepare_data.py
python3 DM_Project_Channel2/channel1_scripts/scripts/prepare_kumo_rfm_files.py
python3 DM_Project_Channel2/channel1_scripts/scripts/prepare_kumo_eval_files.py
```

These scripts generate the relational tables used for the Kumo upload and evaluation workflow, including:

- `kumo_county_table.csv`
- `kumo_fire_event_table.csv`
- `kumo_fire_event_rfm.csv`
- `kumo_fire_event_rfm_hidden.csv`
- `kumo_actual_2020_labels.csv`

### 3. Kumo Prediction Query

The core prediction query used in the experiment was:

```text
PREDICT kumo_fire_event_rfm_hidden.label_next_month
FOR EACH kumo_fire_event_rfm_hidden.row_id
```

This was applied to the Kumo-ready fire-event table, with a relationship to the county table through:

```text
kumo_fire_event_rfm_hidden.county_id -> kumo_county_table.county_id
```

### 4. Evaluation Step

Once predictions were exported, the subset evaluation script was run:

```bash
python3 DM_Project_Channel2/channel1_scripts/scripts/evaluate_kumo_subset.py
```

This script compares the exported Kumo predictions with the observed 2020 labels for the evaluation subset.

### 5. Notes

- The Kumo component was exploratory and not the final production model.
- Results are strongest as a proof-of-concept for relational forecasting rather than a full deployment benchmark.
- Full validation would require complete batch prediction export and a larger evaluation set.

---
>>>>>>> 5205b55e (Add Kumo and project overview documentation)

## Repository Structure

```text
<<<<<<< HEAD
Datamining-class-project/
├── channel1/
├── DM_Project_Channel2/
├── proposedmethod.png
└── .gitignore
=======
Project/
├── channel1/
│   ├── README.md
│   ├── data/
│   ├── results/
│   └── scripts/
├── DM_Project_Channel2/
│   ├── channel1_scripts/
│   ├── channel2_scripts/
│   └── data/
├── FinalReport/
├── README.md
├── kumoenv/
├── .gitignore
├── FinalReport.zip
└── Poster-Template.pdf
```

---

## Notes

- Kumo work is included here as an exploratory component of the project, not as the sole final modeling framework.
- The repository is organized around the wildfire-to-PM2.5 analysis workflow.
- Virtual environments and raw data are not treated as primary repository artifacts.
>>>>>>> 5205b55e (Add Kumo and project overview documentation)
