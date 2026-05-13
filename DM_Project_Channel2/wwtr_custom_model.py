"""
Channel II - Step 4: ML Models for PM2.5 Prediction
Fix: spike metrics use max_pm25 (worst day per month) as ground truth
     so actual spikes are non-zero, and predictions are compared on same scale.
Train: 2015-2018 | Test: 2019-2020
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

script_dir  = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, "venv", "bin", "python")
if os.path.exists(venv_python):
    current = os.path.abspath(sys.executable)
    target  = os.path.abspath(venv_python)
    if current != target:
        os.execv(target, [target] + sys.argv)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, r2_score, accuracy_score,
                              f1_score, classification_report, mean_squared_error)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPIKE_THRESHOLD = 35.4   # EPA 24-hr standard µg/m³

# ── SPIKE METRICS ─────────────────────────────────────────────────────────────
def spike_metrics(y_true_max, y_pred_avg, threshold=SPIKE_THRESHOLD):
    """
    WHY max_pm25 as ground truth:
      Monthly avg_pm25 rarely exceeds 35.4 (averaging dampens peaks).
      But max_pm25 (worst day of month) DOES exceed 35.4 during fire months.
      We flag a month as 'dangerous' if its worst day crossed the EPA limit.

    WHY pred > threshold*0.75 for flagging:
      Models predict next-month avg — not next-month max. A predicted avg of
      ~26 µg/m³ often corresponds to a max day of ~35+ µg/m³.
      Using 75% of threshold on predicted avg aligns the two scales.
    """
    y_true_max = np.array(y_true_max)
    y_pred_avg = np.array(y_pred_avg)

    actual  = y_true_max > threshold          # month had a dangerous day
    flagged = y_pred_avg > threshold * 0.75   # model predicted elevated avg

    tp = int(np.sum( actual &  flagged))
    fn = int(np.sum( actual & ~flagged))      # missed warnings
    fp = int(np.sum(~actual &  flagged))
    tn = int(np.sum(~actual & ~flagged))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2*precision*recall / (precision+recall)
                 if (precision+recall) > 0 else 0.0)

    print(f"  Actual spike months (max>{threshold}): {int(actual.sum())}")
    print(f"  Flagged by model                    : {int(flagged.sum())}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Spike Precision : {precision:.3f}")
    print(f"  Spike Recall    : {recall:.3f}  ← fraction of danger months caught")
    print(f"  Spike F1        : {f1:.3f}")
    print(f"  Missed Warnings : {fn}           ← public health cost")

    return {"Spike_Precision": round(precision,3), "Spike_Recall": round(recall,3),
            "Spike_F1": round(f1,3), "Missed_Warnings": fn,
            "True_Positives": tp, "False_Positives": fp}

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/california_pm25_final_ml_dataset.csv")

FEATURES = ["avg_pm25","max_pm25","std_pm25","days_above_35",
            "pm25_lag1","pm25_lag2","pm25_rolling3","days_above_lag1",
            "month_sin","month_cos"]
REG_TARGET = "next_month_pm25"
CLF_TARGET = "label_next_month_pm25"

train = df[df["year"] <= 2018].dropna(subset=FEATURES+[REG_TARGET])
test  = df[df["year"] >= 2019].dropna(subset=FEATURES+[REG_TARGET])

X_train, y_train_reg = train[FEATURES], train[REG_TARGET]
X_test,  y_test_reg  = test[FEATURES],  test[REG_TARGET]
y_train_clf = train[CLF_TARGET]
y_test_clf  = test[CLF_TARGET]

# Ground truth for spike detection = next month's MAX pm25
# Shift max_pm25 forward by 1 within each county (same logic as next_month_pm25)
df_sorted = df.sort_values(["NAME","year","month"])
df_sorted["next_month_max_pm25"] = df_sorted.groupby("NAME")["max_pm25"].shift(-1)
test_max = df_sorted[df_sorted["year"] >= 2019].dropna(
    subset=FEATURES+[REG_TARGET]).copy()
y_test_max = test_max["next_month_max_pm25"].fillna(test_max["max_pm25"]).values

print(f"Train: {len(train):,} | Test: {len(test):,}")
print(f"Spike months in test (max_pm25 > {SPIKE_THRESHOLD}): "
      f"{int((y_test_max > SPIKE_THRESHOLD).sum())}")

results  = []
xgb_pred = None
rf_pred  = None

# ════════════════════════════════════════════════════════════════════════════
# 1. RANDOM FOREST
# ════════════════════════════════════════════════════════════════════════════
print("\n"+"="*55+"\n--- Random Forest ---")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_reg)
rf_pred = rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test_reg, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test_reg, rf_pred))
rf_r2   = r2_score(y_test_reg, rf_pred)
print(f"MAE={rf_mae:.3f}  RMSE={rf_rmse:.3f}  R²={rf_r2:.3f}")

rf_clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train_clf)
rf_clf_pred = rf_clf.predict(X_test)
print(classification_report(y_test_clf, rf_clf_pred, digits=2))

rf_spike = spike_metrics(y_test_max, rf_pred)
results.append({"Model":"Random Forest","MAE":round(rf_mae,3),"RMSE":round(rf_rmse,3),
                "R2":round(rf_r2,3),
                "Accuracy":round(accuracy_score(y_test_clf,rf_clf_pred),3),
                "F1_class1":round(f1_score(y_test_clf,rf_clf_pred),3), **rf_spike})

# ════════════════════════════════════════════════════════════════════════════
# 2. XGBOOST
# ════════════════════════════════════════════════════════════════════════════
print("\n"+"="*55+"\n--- XGBoost ---")
if XGBOOST_AVAILABLE:
    try:
        xgb_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.8, eval_metric="rmse",
            early_stopping_rounds=30, verbosity=0, random_state=42)
        xgb_model.fit(X_train, y_train_reg,
                      eval_set=[(X_test,y_test_reg)], verbose=False)
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

        xgb_spike = spike_metrics(y_test_max, xgb_pred)
        results.append({"Model":"XGBoost","MAE":round(xgb_mae,3),
                        "RMSE":round(xgb_rmse,3),"R2":round(xgb_r2,3),
                        "Accuracy":round(accuracy_score(y_test_clf,xgb_clf_pred),3),
                        "F1_class1":round(f1_score(y_test_clf,xgb_clf_pred),3),
                        **xgb_spike})
    except Exception as e:
        print(f"XGBoost failed: {e}")
else:
    print("XGBoost not available — run: brew install libomp")

# ════════════════════════════════════════════════════════════════════════════
# 3. LSTM
# ════════════════════════════════════════════════════════════════════════════
print("\n"+"="*55+"\n--- LSTM ---")
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler

    SEQ_LEN  = 3
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc  = scaler_X.transform(X_test)
    y_train_sc = scaler_y.fit_transform(y_train_reg.values.reshape(-1,1)).flatten()

    def make_seq(X, y, s):
        return (np.array([X[i-s:i] for i in range(s,len(X))]),
                np.array([y[i]     for i in range(s,len(y))]))

    Xt, yt = make_seq(X_train_sc, y_train_sc, SEQ_LEN)
    Xv, _  = make_seq(X_test_sc,  np.zeros(len(X_test_sc)), SEQ_LEN)
    print(f"LSTM input shape: {Xt.shape}")

    model = Sequential([LSTM(64,return_sequences=True,
                              input_shape=(SEQ_LEN,len(FEATURES))),
                         Dropout(0.2), LSTM(32), Dropout(0.2), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xt, yt, epochs=50, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(patience=5,restore_best_weights=True)],
              validation_split=0.1)

    lstm_pred = scaler_y.inverse_transform(
                    model.predict(Xv,verbose=0).reshape(-1,1)).flatten()
    yv_actual     = y_test_reg.values[SEQ_LEN:]
    yv_actual_max = y_test_max[SEQ_LEN:]   # aligned max for spike detection

    lstm_mae  = mean_absolute_error(yv_actual, lstm_pred)
    lstm_rmse = np.sqrt(mean_squared_error(yv_actual, lstm_pred))
    lstm_r2   = r2_score(yv_actual, lstm_pred)
    print(f"MAE={lstm_mae:.3f}  RMSE={lstm_rmse:.3f}  R²={lstm_r2:.3f}")

    lstm_spike = spike_metrics(yv_actual_max, lstm_pred)

    # Classification head
    y_clf_arr       = y_train_clf.values.astype(float)
    y_test_clf_arr  = y_test_clf.values.astype(float)
    Xt_c, yt_c = make_seq(X_train_sc, y_clf_arr, SEQ_LEN)
    Xv_c, yv_c = make_seq(X_test_sc,  y_test_clf_arr, SEQ_LEN)
    clf_m = Sequential([LSTM(64,return_sequences=True,
                              input_shape=(SEQ_LEN,len(FEATURES))),
                         Dropout(0.2), LSTM(32), Dropout(0.2),
                         Dense(1,activation="sigmoid")])
    clf_m.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    clf_m.fit(Xt_c, yt_c, epochs=50, batch_size=32, verbose=0,
              callbacks=[EarlyStopping(patience=5,restore_best_weights=True)],
              validation_split=0.1)
    lstm_clf_pred = (clf_m.predict(Xv_c,verbose=0).flatten()>0.5).astype(int)

    results.append({"Model":"LSTM","MAE":round(lstm_mae,3),
                    "RMSE":round(lstm_rmse,3),"R2":round(lstm_r2,3),
                    "Accuracy":round(accuracy_score(yv_c,lstm_clf_pred),3),
                    "F1_class1":round(f1_score(yv_c,lstm_clf_pred),3),
                    **lstm_spike})
except ImportError:
    print("TensorFlow not installed — run: pip install tensorflow")

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results)
cols = ["Model","MAE","RMSE","R2","Accuracy","F1_class1",
        "Spike_Precision","Spike_Recall","Spike_F1",
        "Missed_Warnings","True_Positives","False_Positives"]
results_df = results_df.reindex(columns=[c for c in cols if c in results_df.columns])
print(f"\n{'='*75}\nFULL RESULTS")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUTPUT_DIR}/pm25_model_results.csv", index=False)

pred_vals  = xgb_pred if xgb_pred is not None else rf_pred
pred_model = "XGBoost" if xgb_pred is not None else "Random Forest"
test_out   = test[["NAME","year","month"]].copy()
test_out["pm25_actual"]    = y_test_reg.values
test_out["pm25_predicted"] = pred_vals
test_out.to_csv(f"{OUTPUT_DIR}/pm25_predictions_2019_2020.csv", index=False)
print(f"\nSaved → {OUTPUT_DIR}/pm25_model_results.csv")
print(f"Saved {pred_model} predictions → {OUTPUT_DIR}/pm25_predictions_2019_2020.csv")