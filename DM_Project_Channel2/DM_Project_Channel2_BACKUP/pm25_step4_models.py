"""
Channel II - Step 4: ML Models for PM2.5 Prediction
Input:  california_pm25_final_ml_dataset.csv
Output: pm25_model_results.csv, pm25_predictions_2019_2020.csv

Trains Random Forest, XGBoost, and LSTM — same comparative framework as Channel I.
Task: Regression (predict next-month avg_pm25) + Classification (high PM2.5 label)
Train: 2015–2018 | Test: 2019–2020
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, r2_score,
                              accuracy_score, f1_score,
                              classification_report, mean_squared_error)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/california_pm25_final_ml_dataset.csv")

FEATURES = [
    "avg_pm25", "max_pm25", "std_pm25", "days_above_35",
    "pm25_lag1", "pm25_lag2", "pm25_rolling3", "days_above_lag1",
    "month_sin", "month_cos"
]
REG_TARGET = "next_month_pm25"
CLF_TARGET = "label_next_month_pm25"

train = df[df["year"] <= 2018].dropna(subset=FEATURES + [REG_TARGET])
test  = df[df["year"] >= 2019].dropna(subset=FEATURES + [REG_TARGET])

X_train, y_train_reg = train[FEATURES], train[REG_TARGET]
X_test,  y_test_reg  = test[FEATURES],  test[REG_TARGET]
y_train_clf = train[CLF_TARGET]
y_test_clf  = test[CLF_TARGET]

print(f"Train: {len(train):,} | Test: {len(test):,}")

results = []

# ── 1. RANDOM FOREST (Regression) ───────────────────────────────────────────
print("\n--- Random Forest ---")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_reg)
rf_pred = rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test_reg, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test_reg, rf_pred))
rf_r2   = r2_score(y_test_reg, rf_pred)
print(f"MAE={rf_mae:.3f}  RMSE={rf_rmse:.3f}  R²={rf_r2:.3f}")

# Also as classifier
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train_clf)
rf_clf_pred = rf_clf.predict(X_test)
print(classification_report(y_test_clf, rf_clf_pred, digits=2))
results.append({"Model": "Random Forest",
                "MAE": rf_mae, "RMSE": rf_rmse, "R2": rf_r2,
                "Accuracy": accuracy_score(y_test_clf, rf_clf_pred),
                "F1_class1": f1_score(y_test_clf, rf_clf_pred)})

# ── 2. XGBOOST (Regression) ──────────────────────────────────────────────────
if XGBOOST_AVAILABLE:
    print("\n--- XGBoost ---")
    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8,
            eval_metric="rmse", early_stopping_rounds=30,
            verbosity=0, random_state=42
        )
        xgb_model.fit(X_train, y_train_reg,
                      eval_set=[(X_test, y_test_reg)],
                      verbose=False)
        xgb_pred = xgb_model.predict(X_test)

        xgb_mae  = mean_absolute_error(y_test_reg, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test_reg, xgb_pred))
        xgb_r2   = r2_score(y_test_reg, xgb_pred)
        print(f"MAE={xgb_mae:.3f}  RMSE={xgb_rmse:.3f}  R²={xgb_r2:.3f}")

        xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=6,
                                      use_label_encoder=False, verbosity=0,
                                      random_state=42, eval_metric="logloss")
        xgb_clf.fit(X_train, y_train_clf)
        xgb_clf_pred = xgb_clf.predict(X_test)
        print(classification_report(y_test_clf, xgb_clf_pred, digits=2))
        results.append({"Model": "XGBoost",
                        "MAE": xgb_mae, "RMSE": xgb_rmse, "R2": xgb_r2,
                        "Accuracy": accuracy_score(y_test_clf, xgb_clf_pred),
                        "F1_class1": f1_score(y_test_clf, xgb_clf_pred)})
    except Exception as e:
        print(f"XGBoost training failed: {e}")
else:
    print("\n--- XGBoost ---")
    print("XGBoost not available — skipping. Run: brew install libomp")

# ── 3. LSTM (Regression) ─────────────────────────────────────────────────────
print("\n--- LSTM ---")
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler

    SEQ_LEN = 3   # 3-month sequences (matches Channel I)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc  = scaler_X.transform(X_test)
    y_train_sc = scaler_y.fit_transform(y_train_reg.values.reshape(-1,1)).flatten()
    y_test_sc  = scaler_y.transform(y_test_reg.values.reshape(-1,1)).flatten()

    def make_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i-seq_len:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    Xt, yt = make_sequences(X_train_sc, y_train_sc, SEQ_LEN)
    Xv, yv = make_sequences(X_test_sc,  y_test_sc,  SEQ_LEN)
    print(f"LSTM input shape: {Xt.shape}")

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(FEATURES))),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xt, yt, epochs=50, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
              validation_split=0.1)

    lstm_pred_sc = model.predict(Xv, verbose=0).flatten()
    lstm_pred    = scaler_y.inverse_transform(lstm_pred_sc.reshape(-1,1)).flatten()
    yv_actual    = scaler_y.inverse_transform(yv.reshape(-1,1)).flatten()

    lstm_mae  = mean_absolute_error(yv_actual, lstm_pred)
    lstm_rmse = np.sqrt(mean_squared_error(yv_actual, lstm_pred))
    lstm_r2   = r2_score(yv_actual, lstm_pred)
    print(f"MAE={lstm_mae:.3f}  RMSE={lstm_rmse:.3f}  R²={lstm_r2:.3f}")

    # LSTM Classification
    y_train_clf_sc = scaler_y.fit_transform(y_train_clf.values.reshape(-1,1)).flatten()
    y_test_clf_sc  = scaler_y.transform(y_test_clf.values.reshape(-1,1)).flatten()
    
    Xt_clf, yt_clf = make_sequences(X_train_sc, y_train_clf_sc, SEQ_LEN)
    Xv_clf, yv_clf = make_sequences(X_test_sc,  y_test_clf_sc,  SEQ_LEN)
    
    model_clf = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(FEATURES))),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model_clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model_clf.fit(Xt_clf, yt_clf, epochs=50, batch_size=32, verbose=0,
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                  validation_split=0.1)
    
    lstm_clf_pred_sc = model_clf.predict(Xv_clf, verbose=0).flatten()
    lstm_clf_pred = (lstm_clf_pred_sc > 0.5).astype(int)
    
    lstm_acc = accuracy_score(yv_clf, lstm_clf_pred)
    lstm_f1  = f1_score(yv_clf, lstm_clf_pred)
    
    results.append({"Model": "LSTM", "MAE": lstm_mae,
                    "RMSE": lstm_rmse, "R2": lstm_r2,
                    "Accuracy": lstm_acc, "F1_class1": lstm_f1})

except ImportError:
    print("TensorFlow not installed — skipping LSTM. Run: pip install tensorflow")

# ── SAVE RESULTS ─────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
print(f"\n{'='*55}")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUTPUT_DIR}/pm25_model_results.csv", index=False)

# Save predictions for linking step (use XGBoost if available, else Random Forest)
test_out = test[["NAME","year","month"]].copy()
test_out["pm25_actual"]    = y_test_reg.values
test_out["pm25_predicted"] = xgb_pred if XGBOOST_AVAILABLE else rf_pred
model_used = "XGBoost" if XGBOOST_AVAILABLE else "Random Forest"
test_out.to_csv(f"{OUTPUT_DIR}/pm25_predictions_2019_2020.csv", index=False)
print(f"Saved {model_used} predictions → {OUTPUT_DIR}/pm25_predictions_2019_2020.csv")
