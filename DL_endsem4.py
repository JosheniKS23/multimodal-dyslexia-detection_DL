    # ==============================
# IMPORTS
# ==============================
import os, cv2, torch, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# PATHS (CHANGE IF NEEDED)
# ==============================
train_img_dir = r"C:\Users\joshe\Downloads\synthdata_handwriting\kaggle\working\synthdata\images\train"
train_lbl_dir = r"C:\Users\joshe\Downloads\synthdata_handwriting\kaggle\working\synthdata\labels\train"

val_img_dir = r"C:\Users\joshe\Downloads\synthdata_handwriting\kaggle\working\synthdata\images\val"
val_lbl_dir = r"C:\Users\joshe\Downloads\synthdata_handwriting\kaggle\working\synthdata\labels\val"

# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# ==============================
# DATASET
# ==============================
class CharacterDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label_file = self.files[idx]
        img_path = os.path.join(self.image_dir, label_file.replace(".txt", ".png"))
        label_path = os.path.join(self.label_dir, label_file)

        image = cv2.imread(img_path)
        h, w, _ = image.shape

        patches, labels = [], []

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            cls, x, y, bw, bh = map(float, line.split())
            cls = int(cls)

            if cls == 2:
                continue

            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (32, 32))

            if self.transform:
                crop = self.transform(crop)

            patches.append(crop)
            labels.append(cls)

        return patches, labels

# ==============================
# COLLATE
# ==============================
def collate_fn(batch):
    images, labels = [], []

    for patches, lbls in batch:
        for p, l in zip(patches, lbls):
            images.append(p)
            labels.append(l)

    return torch.stack(images), torch.tensor(labels)

# ==============================
# LOAD DATA
# ==============================
train_loader = DataLoader(
    CharacterDataset(train_img_dir, train_lbl_dir, transform),
    batch_size=8, shuffle=True, collate_fn=collate_fn
)

val_loader = DataLoader(
    CharacterDataset(val_img_dir, val_lbl_dir, transform),
    batch_size=8, shuffle=False, collate_fn=collate_fn
)

# ==============================
# MODEL
# ==============================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN().to(device)

# ==============================
# LOSS (CLASS WEIGHT FIX)
# ==============================
class_weights = torch.tensor([1.0, 2.5]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==============================
# TRAINING
# ==============================
epochs = 5

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")

# ==============================
# VALIDATION + PROBABILITIES
# ==============================
model.eval()
all_probs, all_labels = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# ==============================
# THRESHOLD OPTIMIZATION
# ==============================
thresholds = np.linspace(0, 1, 100)
f1_scores = []

best_f1 = 0
best_thresh = 0

for t in thresholds:
    preds = (all_probs >= t).astype(int)
    f1 = f1_score(all_labels, preds)
    f1_scores.append(f1)

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print("Best Threshold:", best_thresh)
print("Best F1:", best_f1)

# ==============================
# FINAL METRICS
# ==============================
final_preds = (all_probs >= best_thresh).astype(int)

acc = accuracy_score(all_labels, final_preds)
precision = precision_score(all_labels, final_preds)
recall = recall_score(all_labels, final_preds)
f1 = f1_score(all_labels, final_preds)
cm = confusion_matrix(all_labels, final_preds)

print("\nFINAL METRICS")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:\n", cm)

# ==============================
# CONFUSION MATRIX PLOT
# ==============================
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0,1], ["Normal", "Reversal"])
plt.yticks([0,1], ["Normal", "Reversal"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# ROC CURVE
# ==============================
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()

# ==============================
# PRECISION-RECALL CURVE
# ==============================
precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs)

plt.plot(recall_vals, precision_vals)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# ==============================
# F1 vs THRESHOLD (NEW)
# ==============================
plt.plot(thresholds, f1_scores)
plt.axvline(best_thresh, linestyle='--', label=f"Best={best_thresh:.2f}")
plt.title("F1 Score vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.legend()
plt.show()