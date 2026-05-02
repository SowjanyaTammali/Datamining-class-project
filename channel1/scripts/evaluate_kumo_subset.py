import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

actual = pd.read_csv("kumo_actual_2020_labels.csv")
pred = pd.read_csv("kumo_predictions_2020_part1.csv")

df = actual.merge(
    pred,
    on=["row_id", "county_id", "year", "month"],
    how="inner"
)

df["pred_label"] = df["predicted_class"].astype(float).astype(int)

y_true = df["label_next_month"].astype(float).astype(int)
y_pred = df["pred_label"]

print("Evaluated rows:", len(df))
print("Accuracy:", accuracy_score(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))
