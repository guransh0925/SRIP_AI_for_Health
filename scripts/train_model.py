import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_model import CNN1D

# Load dataset
df = pd.read_pickle('Dataset/breathing_dataset.pkl')

# Encode labels
label_map = {'Normal': 0, 'Hypopnea': 1, 'Obstructive Apnea': 2, 'Mixed Apnea': 3, 'Central Apnea': 4}
df['label_enc'] = df['label'].map(label_map).fillna(0).astype(int)
num_classes = df['label_enc'].nunique()

# # CNN Model
# class CNN1D(nn.Module):
#     def __init__(self, input_length, num_classes):
#         super(CNN1D, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(3, 16, kernel_size=7, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(16, 32, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         self.fc = nn.Linear(64, num_classes)

#     def forward(self, x):
#         x = self.net(x)
#         x = x.squeeze(-1)
#         return self.fc(x)

def prepare_input(row):
    flow = np.array(row['flow'])
    thorac = np.array(row['thorac'])
    # resample spo2 to match flow length (960 samples)
    spo2 = np.array(row['spo2'])
    spo2_resampled = np.interp(
        np.linspace(0, 1, len(flow)),
        np.linspace(0, 1, len(spo2)),
        spo2
    )
    return np.stack([flow, thorac, spo2_resampled], axis=0)

participants = df['participant'].unique()
all_preds = []
all_labels = []

for test_participant in participants:
    print(f"\nFold: test on {test_participant}")

    train_df = df[df['participant'] != test_participant].reset_index(drop=True)
    test_df = df[df['participant'] == test_participant].reset_index(drop=True)

    X_train = np.stack([prepare_input(row) for _, row in train_df.iterrows()])
    y_train = train_df['label_enc'].values
    X_test = np.stack([prepare_input(row) for _, row in test_df.iterrows()])
    y_test = test_df['label_enc'].values

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    model = CNN1D(input_length=960, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    # Fill weights for all classes including unseen ones
    full_weights = np.ones(num_classes)
    for i, c in enumerate(classes):
        full_weights[c] = weights[i]
    class_weights = torch.tensor(full_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/10 - Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).numpy()

    all_preds.extend(preds)
    all_labels.extend(y_test)

# Metrics
label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1]) if v < num_classes]
print("\n--- Results ---")
print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
print(f"Precision: {precision_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
print(f"Recall:    {recall_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names[:len(cm)])
disp.plot(xticks_rotation=45)
plt.title('Confusion Matrix - Leave-One-Out CV')
plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/confusion_matrix.png')
plt.show()
print("Confusion matrix saved to results/confusion_matrix.png")