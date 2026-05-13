# Predictive Modeling of Wildfire Activity and PM2.5 Air Quality Trends

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

## Repository Structure

```text
Datamining-class-project/
├── channel1/
├── DM_Project_Channel2/
├── proposedmethod.png
└── .gitignore
