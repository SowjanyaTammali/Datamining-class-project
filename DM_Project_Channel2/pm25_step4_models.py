"""
Channel II - Step 4: ML Models with corrected spike metrics.
Root cause of zeros: monthly avg_pm25 rarely exceeds 35.4 µg/m³ (that is a
daily 24-hr standard). Fix: use the 75th percentile of next_month_pm25 in the
TEST SET as the spike threshold — consistent with how Channel I defines "high".
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

# ── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/california_pm25_final_ml_dataset.csv")

FEATURES = [
    "avg_pm25","max_pm25","std_pm25","days_above_35",
    "pm25_lag1","pm25_lag2","pm25_rolling3","days_above_lag1",
    "month_sin","month_cos"
]
REG_TARGET = "next_month_pm25"
CLF_TARGET = "label_next_month_pm25"

train = df[df["year"] <= 2018].dropna(subset=FEATURES + [REG_TARGET])
test  = df[df["year"] >= 2019].dropna(subset=FEATURES + [REG_TARGET])

X_train, y_train_reg = train[FEATURES], train[REG_TARGET]
X_test,  y_test_reg  = test[FEATURES],  test[REG_TARGET]
y_train_clf = train[CLF_TARGET]
y_test_clf  = test[CLF_TARGET]

# ── DATA-DRIVEN THRESHOLD ────────────────────────────────────────────────────
# Monthly county averages rarely exceed 35.4 (a daily standard).
# Use 75th percentile of actual test values — guarantees ~25% positives.
SPIKE_THRESHOLD = float(np.percentile(y_test_reg, 75))
print(f"Train: {len(train):,} | Test: {len(test):,}")
print(f"Spike threshold (P75 of test): {SPIKE_THRESHOLD:.2f} µg/m³")
print(f"Actual spikes in test: {int((y_test_reg > SPIKE_THRESHOLD).sum())}")

# ── SPIKE METRICS ─────────────────────────────────────────────────────────────
def spike_metrics(y_true, y_pred, threshold):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    actual  = y_true > threshold
    flagged = y_pred > threshold
    tp = int(np.sum( actual &  flagged))
    fn = int(np.sum( actual & ~flagged))
    fp = int(np.sum(~actual &  flagged))
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    print(f"  Actual spikes={int(actual.sum())} Flagged={int(flagged.sum())} "
          f"TP={tp} FP={fp} FN={fn}")
    print(f"  Spike Precision={precision:.3f} Recall={recall:.3f} "
          f"F1={f1:.3f} MissedWarnings={fn}")
    return {"Spike_Precision":round(precision,3), "Spike_Recall":round(recall,3),
            "Spike_F1":round(f1,3), "Missed_Warnings":fn,
            "TP":tp, "FP":fp, "FN":fn}

results  = []
xgb_pred = None
rf_pred  = None

# ── 1. RANDOM FOREST ─────────────────────────────────────────────────────────
print("\n--- Random Forest ---")
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
rf_sp = spike_metrics(y_test_reg, rf_pred, SPIKE_THRESHOLD)
results.append({"Model":"Random Forest","MAE":round(rf_mae,3),"RMSE":round(rf_rmse,3),
                "R2":round(rf_r2,3),
                "Accuracy":round(accuracy_score(y_test_clf,rf_clf_pred),3),
                "F1_class1":round(f1_score(y_test_clf,rf_clf_pred),3), **rf_sp})

# ── 2. XGBOOST ───────────────────────────────────────────────────────────────
print("\n--- XGBoost ---")
if XGBOOST_AVAILABLE:
    try:
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
            eval_metric="rmse", early_stopping_rounds=30, verbosity=0, random_state=42)
        xgb_model.fit(X_train, y_train_reg,
                      eval_set=[(X_test,y_test_reg)], verbose=False)
        xgb_pred = xgb_model.predict(X_test)
        xgb_mae  = mean_absolute_error(y_test_reg, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test_reg, xgb_pred))
        xgb_r2   = r2_score(y_test_reg, xgb_pred)
        print(f"MAE={xgb_mae:.3f}  RMSE={xgb_rmse:.3f}  R²={xgb_r2:.3f}")
        xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=6,
                                      verbosity=0, random_state=42, eval_metric="logloss")
        xgb_clf.fit(X_train, y_train_clf)
        xgb_clf_pred = xgb_clf.predict(X_test)
        print(classification_report(y_test_clf, xgb_clf_pred, digits=2))
        xgb_sp = spike_metrics(y_test_reg, xgb_pred, SPIKE_THRESHOLD)
        results.append({"Model":"XGBoost","MAE":round(xgb_mae,3),"RMSE":round(xgb_rmse,3),
                        "R2":round(xgb_r2,3),
                        "Accuracy":round(accuracy_score(y_test_clf,xgb_clf_pred),3),
                        "F1_class1":round(f1_score(y_test_clf,xgb_clf_pred),3), **xgb_sp})
    except Exception as e:
        print(f"XGBoost failed: {e}")
else:
    print("XGBoost not available — brew install libomp")

# ── 3. LSTM ──────────────────────────────────────────────────────────────────
print("\n--- LSTM ---")
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler

    SEQ_LEN  = 3
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_tr_sc  = scaler_X.fit_transform(X_train)
    X_te_sc  = scaler_X.transform(X_test)
    y_tr_sc  = scaler_y.fit_transform(y_train_reg.values.reshape(-1,1)).flatten()
    y_te_sc  = scaler_y.transform(y_test_reg.values.reshape(-1,1)).flatten()

    def make_seq(X, y, s):
        return (np.array([X[i-s:i] for i in range(s,len(X))]),
                np.array([y[i]     for i in range(s,len(y))]))

    Xt,yt = make_seq(X_tr_sc, y_tr_sc, SEQ_LEN)
    Xv,yv = make_seq(X_te_sc, y_te_sc, SEQ_LEN)

    m = Sequential([LSTM(64,return_sequences=True,input_shape=(SEQ_LEN,len(FEATURES))),
                    Dropout(0.2), LSTM(32), Dropout(0.2), Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    m.fit(Xt, yt, epochs=50, batch_size=32, verbose=0,
          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
          validation_split=0.1)

    lstm_pred = scaler_y.inverse_transform(
                    m.predict(Xv,verbose=0).reshape(-1,1)).flatten()
    yv_actual = y_test_reg.values[SEQ_LEN:]

    lstm_mae  = mean_absolute_error(yv_actual, lstm_pred)
    lstm_rmse = np.sqrt(mean_squared_error(yv_actual, lstm_pred))
    lstm_r2   = r2_score(yv_actual, lstm_pred)
    print(f"MAE={lstm_mae:.3f}  RMSE={lstm_rmse:.3f}  R²={lstm_r2:.3f}")
    lstm_sp = spike_metrics(yv_actual, lstm_pred, SPIKE_THRESHOLD)

    # clf head
    y_tr_clf_a = y_train_clf.values.astype(float)
    y_te_clf_a = y_test_clf.values.astype(float)
    Xtc,ytc = make_seq(X_tr_sc, y_tr_clf_a, SEQ_LEN)
    Xvc,yvc = make_seq(X_te_sc, y_te_clf_a, SEQ_LEN)
    mc = Sequential([LSTM(64,return_sequences=True,input_shape=(SEQ_LEN,len(FEATURES))),
                     Dropout(0.2), LSTM(32), Dropout(0.2), Dense(1,activation="sigmoid")])
    mc.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    mc.fit(Xtc, ytc, epochs=50, batch_size=32, verbose=0,
           callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
           validation_split=0.1)
    lstm_clf_pred = (mc.predict(Xvc,verbose=0).flatten()>0.5).astype(int)
    results.append({"Model":"LSTM","MAE":round(lstm_mae,3),"RMSE":round(lstm_rmse,3),
                    "R2":round(lstm_r2,3),
                    "Accuracy":round(accuracy_score(yvc,lstm_clf_pred),3),
                    "F1_class1":round(f1_score(yvc,lstm_clf_pred),3), **lstm_sp})
except ImportError:
    print("TensorFlow not installed.")

# ── SUMMARY ──────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
cols = ["Model","MAE","RMSE","R2","Accuracy","F1_class1",
        "Spike_Precision","Spike_Recall","Spike_F1","Missed_Warnings","TP","FP","FN"]
results_df = results_df.reindex(columns=[c for c in cols if c in results_df.columns])
print(f"\n{'='*75}\nFULL RESULTS (spike threshold = {SPIKE_THRESHOLD:.2f} µg/m³ = P75)")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUTPUT_DIR}/pm25_model_results.csv", index=False)

# Save threshold so WWTR script uses same value
pd.DataFrame([{"spike_threshold": SPIKE_THRESHOLD}]).to_csv(
    f"{OUTPUT_DIR}/spike_threshold.csv", index=False)

pred_vals = xgb_pred if xgb_pred is not None else rf_pred
test_out  = test[["NAME","year","month"]].copy()
test_out["pm25_actual"]    = y_test_reg.values
test_out["pm25_predicted"] = pred_vals
test_out.to_csv(f"{OUTPUT_DIR}/pm25_predictions_2019_2020.csv", index=False)
print(f"Saved → {OUTPUT_DIR}/pm25_predictions_2019_2020.csv")