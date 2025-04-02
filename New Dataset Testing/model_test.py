import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch.nn as nn

# Updated MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Load Dataset to get feature shape and scaler
data = pd.read_csv('dataset.csv')
X = data.drop('label', axis=1).values

# Load and fit the same scaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Load Model
model = MLP(X.shape[1])  # Ensure model expects 37 features
model.load_state_dict(torch.load('best_mlp_model.pth'))
model.eval()

# Load New Inputs from Text File
with open('test.txt', 'r') as file:
    new_inputs = [list(map(float, line.strip().split(','))) for line in file]

# Convert to numpy array
new_inputs = np.array(new_inputs)

# Ensure 37 features by removing the extra column if present
if new_inputs.shape[1] != X.shape[1]:
    new_inputs = new_inputs[:, :X.shape[1]]  # Trim extra column

# Normalize and convert to tensor
new_inputs = scaler.transform(new_inputs)
new_inputs_tensor = torch.tensor(new_inputs, dtype=torch.float32)

# Prediction
with torch.no_grad():
    outputs = model(new_inputs_tensor).squeeze()
    predicted_labels = (outputs > 0.5).int().tolist()

# Display Results
for i, label in enumerate(predicted_labels):
    print(f"Row {i+1}: Predicted Label - {label}")
