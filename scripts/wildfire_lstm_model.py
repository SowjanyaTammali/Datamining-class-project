import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load data
X = np.load("X_lstm.npy")
y = np.load("y_lstm.npy")

# -------- NORMALIZE FEATURES --------
scaler = StandardScaler()
X_flat = X.reshape(-1, X.shape[2])
X_scaled = scaler.fit_transform(X_flat)
X = X_scaled.reshape(X.shape)

# Convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Train/test split (80/20 time-based)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------- HANDLE IMBALANCE --------
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# -------- MODEL --------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out  # raw logits (NO sigmoid)

model = LSTMModel(input_size=X.shape[2], hidden_size=32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------- TRAIN --------
epochs = 25

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------- EVALUATE --------
model.eval()
with torch.no_grad():
    test_logits = model(X_test).squeeze()
    test_probs = torch.sigmoid(test_logits)
    predictions = (test_probs > 0.5).int().numpy()
    y_true = y_test.int().numpy()

print("\nAccuracy:", accuracy_score(y_true, predictions))
print("\nClassification Report:\n")
print(classification_report(y_true, predictions))