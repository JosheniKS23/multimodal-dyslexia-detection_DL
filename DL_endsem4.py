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
# MODEL (3 CLASS + FEATURE EXTRACTOR)
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

        # feature layer
        self.feature_fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU()
        )

        # classifier
        self.classifier = nn.Linear(128, 3)

    def forward(self, x, return_features=False):
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        features = self.feature_fc(x)

        if return_features:
            return features

        out = self.classifier(features)
        return out


model = SimpleCNN().to(device)

# ==============================
# LOSS (3 CLASS)
# ==============================
class_weights = torch.tensor([
    1.0,  # normal
    2.5,  # reversal
    2.0   # corrected
]).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==============================
# TRAINING
# ==============================
epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0

    print(f"\nEpoch {epoch+1}")

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Loss: {train_loss:.4f}")

# ==============================
# VALIDATION + METRICS (3 CLASS)
# ==============================
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(
            probs,
            dim=1
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

all_probs = np.array(all_probs)
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ==============================
# BASIC METRICS
# ==============================
acc = accuracy_score(
    all_labels,
    all_preds
)

precision = precision_score(
    all_labels,
    all_preds,
    average='weighted',
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_preds,
    average='weighted',
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_preds,
    average='weighted',
    zero_division=0
)

print("\nFINAL METRICS")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ==============================
# CLASSIFICATION REPORT
# ==============================
print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[
            "Normal",
            "Reversal",
            "Corrected"
        ],
    zero_division=0
    )
)

# ==============================
# CONFUSION MATRIX
# ==============================
cm = confusion_matrix(
    all_labels,
    all_preds
)
print("\nPer-class Accuracy")

classes = [
    "Normal",
    "Reversal",
    "Corrected"
]

for i, cls_name in enumerate(classes):

    total_class_samples = cm[i].sum()

    if total_class_samples == 0:
        class_acc = 0
    else:
        class_acc = (
            cm[i, i]
            / total_class_samples
        )

    print(
        f"{cls_name}: "
        f"{class_acc:.4f}"
    )

plt.figure(figsize=(7,6))

plt.imshow(cm, interpolation='nearest')

plt.title(
    "Confusion Matrix"
)

plt.colorbar()

classes = [
    "Normal",
    "Reversal",
    "Corrected"
]

plt.xticks(
    range(3),
    classes
)

plt.yticks(
    range(3),
    classes
)

for i in range(3):
    for j in range(3):
        plt.text(
            j,
            i,
            cm[i, j],
            ha='center',
            va='center'
        )

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# ROC CURVE (ONE VS REST)
# ==============================
binary_labels = label_binarize(
    all_labels,
    classes=[0,1,2]
)

plt.figure(figsize=(7,6))

class_names = [
    "Normal",
    "Reversal",
    "Corrected"
]

for i in range(3):

    fpr, tpr, _ = roc_curve(
        binary_labels[:, i],
        all_probs[:, i]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{class_names[i]} AUC = {roc_auc:.3f}"
    )

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Multi-Class ROC Curve"
)

plt.legend()
plt.show()

# ==============================
# PRECISION-RECALL CURVE
# ==============================
plt.figure(figsize=(7,6))

for i in range(3):

    precision_vals, recall_vals, _ = precision_recall_curve(
        binary_labels[:, i],
        all_probs[:, i]
    )

    plt.plot(
        recall_vals,
        precision_vals,
        label=class_names[i]
    )

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve"
)

plt.legend()
plt.show()

# ==============================
# MULTI-CLASS AUC
# ==============================
multi_auc = roc_auc_score(
    binary_labels,
    all_probs,
    multi_class='ovr'
)

print(
    f"\nMulti-class AUC: {multi_auc:.4f}"
)
# ==============================
# SAVE MODEL WEIGHTS
# ==============================
torch.save(
    model.state_dict(),
    "cnn_image_branch.pth"
)

print("✅ Image model saved")

# ==============================
# FEATURE EXTRACTION
# ==============================
model.eval()

image_features = []
image_labels = []
feature_loader = DataLoader(
    CharacterDataset(
        train_img_dir,
        train_lbl_dir,
        transform
    ),
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn
)

with torch.no_grad():
    for images, labels in feature_loader:

        images = images.to(device)

        feats = model(
            images,
            return_features=True
        )

        image_features.append(
            feats.cpu()
        )

        image_labels.append(labels)

image_features = torch.cat(
    image_features
)

image_labels = torch.cat(
    image_labels
)

print(image_features.shape)
print(image_labels.shape)

# ==============================
# MULTIMODAL LABEL MAPPING
# ==============================
# Normal = 0
# Reversal = 1
# Corrected = 1

fusion_labels = image_labels.clone()

fusion_labels[
    fusion_labels == 2
] = 1

# ==============================
# SAVE FEATURES + LABELS
# ==============================
torch.save(
    image_features,
    "image_branch_features.pth"
)

torch.save(
    fusion_labels,
    "image_labels.pth"
)
torch.save(
    image_labels,
    "image_labels_original.pth"
)

print("✅ Features saved")
print("✅ Labels saved")

