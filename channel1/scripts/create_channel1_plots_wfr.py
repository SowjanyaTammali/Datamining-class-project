import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc


# --------------------------------------------------
# Helper: find project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

possible_prediction_files = [
    os.path.join(BASE_DIR, "scripts", "wfr_xgb_predictions.csv"),
    os.path.join(BASE_DIR, "wfr_xgb_predictions.csv"),
    os.path.join(RESULTS_DIR, "wfr_xgb_predictions.csv"),
]

pred_file = None
for path in possible_prediction_files:
    if os.path.exists(path):
        pred_file = path
        break

if pred_file is None:
    raise FileNotFoundError(
        "Could not find wfr_xgb_predictions.csv. "
        "Run wildfire_model_wfr_xgb.py first."
    )

print("Using prediction file:", pred_file)

pred_df = pd.read_csv(pred_file)

# --------------------------------------------------
# 1. Model comparison bar chart
# --------------------------------------------------

model_results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Balanced RF",
        "XGBoost",
        "LSTM",
        "WFR-XGB"
    ],
    "Accuracy": [0.78, 0.78, 0.76, 0.65, 0.7002],
    "Class 1 Recall": [0.40, 0.38, 0.57, 0.64, 0.7287],
    "Class 1 F1": [0.52, 0.50, 0.58, 0.46, 0.5887]
})

model_results.to_csv(
    os.path.join(RESULTS_DIR, "channel1_model_comparison_wfr.csv"),
    index=False
)

x = np.arange(len(model_results["Model"]))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar(x - width, model_results["Accuracy"], width, label="Accuracy")
plt.bar(x, model_results["Class 1 Recall"], width, label="Class 1 Recall")
plt.bar(x + width, model_results["Class 1 F1"], width, label="Class 1 F1")

plt.xticks(x, model_results["Model"], rotation=25, ha="right")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Channel 1 Model Performance Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "channel1_model_comparison_wfr_bar.png"),
    dpi=300
)
plt.close()


# --------------------------------------------------
# 2. Confusion matrix for WFR-XGB
# --------------------------------------------------

y_true = pred_df["label_next_month"].astype(int)
y_pred = pred_df["wfr_xgb_pred"].astype(int)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("WFR-XGB Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["Normal", "High Fire"])
plt.yticks([0, 1], ["Normal", "High Fire"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

plt.tight_layout()
plt.savefig(
    os.path.join(RESULTS_DIR, "wfr_xgb_confusion_matrix.png"),
    dpi=300
)
plt.close()


# --------------------------------------------------
# 3. ROC curve for WFR-XGB
# --------------------------------------------------

if "wfr_xgb_prob_1" in pred_df.columns:
    y_prob = pred_df["wfr_xgb_prob_1"]

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("WFR-XGB ROC Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, "wfr_xgb_roc_curve.png"),
        dpi=300
    )
    plt.close()


# --------------------------------------------------
# 4. Feature importance plot
# Use saved WFR-XGB feature importance if available
# --------------------------------------------------

possible_importance_files = [
    os.path.join(BASE_DIR, "scripts", "wfr_xgb_feature_importance.csv"),
    os.path.join(BASE_DIR, "wfr_xgb_feature_importance.csv"),
    os.path.join(RESULTS_DIR, "wfr_xgb_feature_importance.csv"),
]

importance_file = None
for path in possible_importance_files:
    if os.path.exists(path):
        importance_file = path
        break

if importance_file is not None:
    feature_importance = pd.read_csv(importance_file)

    # Handles both possible formats:
    # Feature,importance OR unnamed index,importance
    if "Feature" not in feature_importance.columns:
        first_col = feature_importance.columns[0]
        feature_importance = feature_importance.rename(columns={first_col: "Feature"})

    if "importance" in feature_importance.columns:
        feature_importance = feature_importance.rename(columns={"importance": "Importance"})

else:
    # Fallback using your printed WFR-XGB gain values
    feature_importance = pd.DataFrame({
        "Feature": [
            "fire_count",
            "month_cos",
            "month_sin",
            "fire_rolling_3",
            "avg_frp",
            "fire_last_1",
            "fire_last_2",
            "frp_last_1"
        ],
        "Importance": [
            12.602584,
            5.643953,
            4.751335,
            4.569623,
            4.265253,
            3.873966,
            3.843816,
            3.731561
        ]
    })

feature_importance = feature_importance.sort_values("Importance", ascending=True)

plt.figure(figsize=(7, 4))
plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.xlabel("Gain Importance")
plt.title("WFR-XGB Feature Importance")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "wfr_xgb_feature_importance_plot.png"),
    dpi=300
)
plt.close()


# --------------------------------------------------
# 5. Threshold tuning plot
# Use saved threshold file if available
# --------------------------------------------------

possible_threshold_files = [
    os.path.join(BASE_DIR, "scripts", "wfr_xgb_threshold_tuning.csv"),
    os.path.join(BASE_DIR, "wfr_xgb_threshold_tuning.csv"),
    os.path.join(RESULTS_DIR, "wfr_xgb_threshold_tuning.csv"),
]

threshold_file = None
for path in possible_threshold_files:
    if os.path.exists(path):
        threshold_file = path
        break

if threshold_file is not None:
    threshold_df = pd.read_csv(threshold_file)

    plt.figure(figsize=(7, 4))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1-score")
    plt.axvline(0.31, linestyle="--", label="Selected threshold = 0.31")

    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tuning for WFR-XGB")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, "wfr_xgb_threshold_tuning_plot.png"),
        dpi=300
    )
    plt.close()


print("\nSaved plots in:", RESULTS_DIR)
print("Generated files:")
print("1. channel1_model_comparison_wfr_bar.png")
print("2. wfr_xgb_confusion_matrix.png")
print("3. wfr_xgb_roc_curve.png")
print("4. wfr_xgb_feature_importance_plot.png")
print("5. wfr_xgb_threshold_tuning_plot.png")
print("6. channel1_model_comparison_wfr.csv")
