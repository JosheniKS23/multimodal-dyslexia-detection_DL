import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

IMAGE_DIR = r"C:\Users\joshe\Downloads\synthdata_handwriting\kaggle\working\synthdata\images\train"
LABEL_DIR = r"C:\Users\joshe\Downloads\synthdata_handwriting\kaggle\working\synthdata\images\train"

OUTPUT_DIR = r"C:\Users\joshe\OneDrive - Amrita vishwa vidyapeetham\_collegefiles\SEM_files_\Sem_4\DL\Endsem_project\processed"

os.makedirs(os.path.join(OUTPUT_DIR, "0"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "1"), exist_ok=True)

count_0, count_1 = 0, 0

for file in os.listdir(IMAGE_DIR):
    if not file.endswith((".png", ".jpg")):
        continue

    img_path = os.path.join(IMAGE_DIR, file)
    label_path = os.path.join(LABEL_DIR, file.replace(".png", ".txt").replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    h, w, _ = image.shape

    with open(label_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        data = line.strip().split()
        class_id = int(data[0])

        if class_id not in [0, 1]:
            continue

        x, y, bw, bh = map(float, data[1:])

        # YOLO → pixel
        x *= w
        y *= h
        bw *= w
        bh *= h

        x1 = int(max(0, x - bw / 2))
        y1 = int(max(0, y - bh / 2))
        x2 = int(min(w, x + bw / 2))
        y2 = int(min(h, y + bh / 2))

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (224, 224))
        _, crop = cv2.threshold(crop, 127, 255, cv2.THRESH_BINARY)

        save_path = os.path.join(OUTPUT_DIR, str(class_id), f"{file}_{i}.png")
        cv2.imwrite(save_path, crop)

        if class_id == 0:
            count_0 += 1
        else:
            count_1 += 1

print("Cropping Done")
print("Class 0:", count_0, "| Class 1:", count_1)

DATA_DIR = OUTPUT_DIR

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Total samples:", len(dataset))


def get_resnet18():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def get_mobilenet():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model.to(device)


def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch + 1} | Loss: {train_loss:.3f} | Val Acc: {val_acc:.3f}")

    return model


def evaluate_model(model, test_loader, name="Model"):
    model.eval()
    preds, labels_all = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())
            labels_all.extend(labels.numpy())

    acc = accuracy_score(labels_all, preds)
    precision = precision_score(labels_all, preds)
    recall = recall_score(labels_all, preds)
    f1 = f1_score(labels_all, preds)

    print(f"\n{name} Performance")
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    cm = confusion_matrix(labels_all, preds)
    plt.imshow(cm)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    plt.show()

    # Metric Plot
    plt.figure()
    plt.bar(["Precision", "Recall", "F1"], [precision, recall, f1])
    plt.title(f"{name} Metrics")
    plt.ylim(0, 1)
    plt.show()

    return acc, precision, recall, f1


print("Training ResNet18...")
resnet = get_resnet18()
resnet = train_model(resnet, train_loader, val_loader)
res_acc, res_p, res_r, res_f1 = evaluate_model(resnet, test_loader, "ResNet18")

print("\nTraining MobileNetV2...")
mobilenet = get_mobilenet()
mobilenet = train_model(mobilenet, train_loader, val_loader)
mob_acc, mob_p, mob_r, mob_f1 = evaluate_model(mobilenet, test_loader, "MobileNetV2")

models = ["ResNet18", "MobileNetV2"]

plt.figure()
x = np.arange(len(models))

plt.bar(x - 0.2, [res_p, mob_p], 0.2, label="Precision")
plt.bar(x, [res_r, mob_r], 0.2, label="Recall")
plt.bar(x + 0.2, [res_f1, mob_f1], 0.2, label="F1")

plt.xticks(x, models)
plt.legend()
plt.title("Model Comparison")
plt.show()