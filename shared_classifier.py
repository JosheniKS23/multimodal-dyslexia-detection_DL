import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)

# ==========================================
# DEVICE
# ==========================================
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)

# ==========================================
# LOAD FEATURES + LABELS
# ==========================================
print("\nLoading .pth files...")

audio_features = torch.load(
    "audio_branch_features.pth"
)

audio_labels = torch.load(
    "audio_labels.pth"
)

image_features = torch.load(
    "image_branch_features.pth"
)

image_labels = torch.load(
    "image_labels.pth"
)

print("\nLoaded Shapes")
print("Audio Features:", audio_features.shape)
print("Audio Labels:", audio_labels.shape)
print("Image Features:", image_features.shape)
print("Image Labels:", image_labels.shape)

# ==========================================
# IMAGE LABEL MAPPING
# corrected -> dyslexia
# Normal = 0
# Reversal = 1
# Corrected = 1
# ==========================================
image_labels_binary = (
    image_labels.clone()
)

image_labels_binary[
    image_labels_binary == 2
] = 1

# ==========================================
# LABEL-CONSISTENT PAIRING
# ==========================================
audio_0_idx = torch.where(
    audio_labels == 0
)[0]

audio_1_idx = torch.where(
    audio_labels == 1
)[0]

image_0_idx = torch.where(
    image_labels_binary == 0
)[0]

image_1_idx = torch.where(
    image_labels_binary == 1
)[0]

num_0 = min(
    len(audio_0_idx),
    len(image_0_idx)
)

num_1 = min(
    len(audio_1_idx),
    len(image_1_idx)
)

print("\nSamples Used")
print("Normal:", num_0)
print("Dyslexia:", num_1)

# ==========================================
# RANDOM LABEL-CONSISTENT PAIRING
# ==========================================
torch.manual_seed(42)

audio_0_idx = audio_0_idx[
    torch.randperm(
        len(audio_0_idx)
    )[:num_0]
]

audio_1_idx = audio_1_idx[
    torch.randperm(
        len(audio_1_idx)
    )[:num_1]
]

image_0_idx = image_0_idx[
    torch.randperm(
        len(image_0_idx)
    )[:num_0]
]

image_1_idx = image_1_idx[
    torch.randperm(
        len(image_1_idx)
    )[:num_1]
]

# ==========================================
# CREATE PAIRED FEATURES
# ==========================================
paired_audio = torch.cat([
    audio_features[audio_0_idx],
    audio_features[audio_1_idx]
])

paired_image = torch.cat([
    image_features[image_0_idx],
    image_features[image_1_idx]
])

fusion_labels = torch.cat([
    torch.zeros(num_0),
    torch.ones(num_1)
])

# ==========================================
# FEATURE CONCATENATION
# ==========================================
fusion_features = torch.cat(
    [
        paired_audio,
        paired_image
    ],
    dim=1
)

print(
    "\nFusion Shape:",
    fusion_features.shape
)

# ==========================================
# DATASET
# ==========================================
dataset = TensorDataset(
    fusion_features,
    fusion_labels
)

train_size = int(
    0.8 * len(dataset)
)

test_size = (
    len(dataset)
    - train_size
)

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# ==========================================
# SHARED CLASSIFIER
# ==========================================
class SharedClassifier(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                1
            )
        )

    def forward(
        self,
        x
    ):
        return self.net(x)

model = SharedClassifier().to(
    device
)

criterion = (
    nn.BCEWithLogitsLoss()
)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# ==========================================
# TRAINING
# ==========================================
epochs = 10

for epoch in range(
    epochs
):

    model.train()

    running_loss = 0

    for feats, labels in train_loader:

        feats = feats.to(
            device
        )

        labels = labels.float().to(
            device
        )

        optimizer.zero_grad()

        outputs = model(
            feats
        ).squeeze()

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = (
        running_loss
        /
        len(train_loader)
    )

    print(
        f"Epoch "
        f"{epoch+1}"
        f"/{epochs}"
        f" Loss: "
        f"{avg_loss:.4f}"
    )

# ==========================================
# EVALUATION
# ==========================================
model.eval()

all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():

    for feats, labels in test_loader:

        feats = feats.to(
            device
        )

        outputs = model(
            feats
        ).squeeze()

        probs = torch.sigmoid(
            outputs
        )

        preds = (
            probs > 0.5
        ).float()

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

# ==========================================
# METRICS
# ==========================================
acc = accuracy_score(
    all_labels,
    all_preds
)

precision = precision_score(
    all_labels,
    all_preds
)

recall = recall_score(
    all_labels,
    all_preds
)

f1 = f1_score(
    all_labels,
    all_preds
)

auc_score = roc_auc_score(
    all_labels,
    all_probs
)

print("\nFINAL RESULTS")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc_score:.4f}")

print("\nClassification Report")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=[
            "Normal",
            "Dyslexia"
        ]
    )
)

# ==========================================
# CONFUSION MATRIX
# ==========================================
cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(6,5))
plt.imshow(cm)
plt.title(
    "Shared Classifier Confusion Matrix"
)
plt.colorbar()

plt.xticks(
    [0,1],
    ["Normal", "Dyslexia"]
)

plt.yticks(
    [0,1],
    ["Normal", "Dyslexia"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================
# ROC CURVE
# ==========================================
fpr, tpr, _ = roc_curve(
    all_labels,
    all_probs
)

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC={auc_score:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Shared Classifier ROC Curve"
)

plt.legend()
plt.show()

# ==========================================
# SAVE MODEL
# ==========================================
torch.save(
    model.state_dict(),
    "shared_classifier.pth"
)

print(
    "\n✅ shared_classifier.pth saved"
)