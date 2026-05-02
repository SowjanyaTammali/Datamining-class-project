import pandas as pd
import numpy as np

df = pd.read_csv("california_county_fire_final_ml_dataset.csv")
df = df.sort_values(by=["NAME", "year", "month"])

features = [
    "fire_count",
    "avg_frp",
    "fire_last_1",
    "fire_last_2",
    "frp_last_1",
    "month_sin",
    "month_cos",
    "fire_rolling_3"
]

sequence_length = 3

X_sequences = []
y_labels = []

for county in df["NAME"].unique():
    county_df = df[df["NAME"] == county].reset_index(drop=True)
    
    for i in range(len(county_df) - sequence_length):
        seq = county_df.loc[i:i+sequence_length-1, features].values
        label = county_df.loc[i+sequence_length-1, "label_next_month"]
        
        X_sequences.append(seq)
        y_labels.append(label)

X = np.array(X_sequences)
y = np.array(y_labels)

np.save("X_lstm.npy", X)
np.save("y_lstm.npy", y)

print("X shape:", X.shape)
print("y shape:", y.shape)