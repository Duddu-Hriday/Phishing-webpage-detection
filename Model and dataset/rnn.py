import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Load Dataset
data = pd.read_csv('dataset.csv')
X = data.drop('label', axis=1).values
y = data['label'].values

# Data Normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch Dataloaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                              torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                             torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding sequence length dimension (batch_size, 1, input_dim)
        _, (hidden, _) = self.lstm(x)  # Get last hidden state
        out = self.fc(hidden[-1])  # Use last layer's hidden state
        return self.sigmoid(out)

# Model Initialization
model = RNNModel(X_train.shape[1])
criterion = nn.BCELoss()  # Binary Cross-Entropy for classification
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training Function
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50):
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation Step
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs).view(-1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss/len(test_loader):.4f} | Val Acc: {val_acc:.4f}")

        # Save Model if Accuracy Improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_rnn_model.pth')
            print("Model Saved with Improved Validation Accuracy")

# Train the Model
train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50)

# Test Evaluation
def evaluate_model(model, test_loader):
    model.load_state_dict(torch.load('best_rnn_model.pth'))
    model.eval()

    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).view(-1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"🔍 Test Loss: {test_loss/len(test_loader):.4f} | Test Accuracy: {test_acc:.4f}")

# Evaluate on Test Set
evaluate_model(model, test_loader)
