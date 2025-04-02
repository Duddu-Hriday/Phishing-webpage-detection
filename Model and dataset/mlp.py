import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

# Updated MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),  # New Layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# Model Initialization
model = MLP(X_train.shape[1])
criterion = nn.BCELoss()  # Binary Cross-Entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training Function
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50):
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # outputs = model(inputs).squeeze()
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
                # outputs = model(inputs).squeeze()
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
            torch.save(model.state_dict(), 'best_mlp_model.pth')
            print("Model Saved with Improved Validation Accuracy")

# Train the Model
train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50)

# Test Evaluation
# def evaluate_model(model, test_loader):
#     model.load_state_dict(torch.load('best_mlp_model.pth'))
#     model.eval()

#     test_loss, correct, total = 0, 0, 0
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             # outputs = model(inputs).squeeze()
#             outputs = model(inputs).view(-1)
#             loss = criterion(outputs, labels)
#             test_loss += loss.item()
#             predicted = (outputs > 0.5).float()
#             correct += (predicted == labels).sum().item()
#             total += labels.size(0)

#     test_acc = correct / total
#     print(f"🔍 Test Loss: {test_loss/len(test_loader):.4f} | Test Accuracy: {test_acc:.4f}")

# # Evaluate on Test Set
# evaluate_model(model, test_loader)


# Test Evaluation with Confusion Matrix
def evaluate_model_with_metrics(model, test_loader):
    model.load_state_dict(torch.load('best_mlp_model.pth'))
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).view(-1)
            predicted = (outputs > 0.5).float()

            all_labels.extend(labels.numpy()) 
            all_predictions.extend(predicted.numpy()) 

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Compute Metrics
    cm = confusion_matrix(all_labels, all_predictions)
    acc = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=1)
    recall = recall_score(all_labels, all_predictions, zero_division=1)
    f1 = f1_score(all_labels, all_predictions)

    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Evaluate with metrics
evaluate_model_with_metrics(model, test_loader)
