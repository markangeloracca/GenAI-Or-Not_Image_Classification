import os
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------

DATASET_PATH = "./dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# TRANSFORMS
# -----------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -----------------------------
# LOAD DATASET
# -----------------------------

full_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)

class_names = full_dataset.classes
print("Classes:", class_names)

# Split dataset
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

val_dataset.dataset.transform = test_transform
test_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -----------------------------
# LOAD RESNET50
# -----------------------------

model = models.resnet50(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 2)
)

model = model.to(device)

# -----------------------------
# LOSS + OPTIMIZER
# -----------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

# -----------------------------
# TRAIN FUNCTION
# -----------------------------

def train_epoch(loader):

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs,1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total

    return total_loss / len(loader), acc


# -----------------------------
# VALIDATION
# -----------------------------

def evaluate(loader):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs,1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total

    return total_loss / len(loader), acc


# -----------------------------
# TRAIN LOOP
# -----------------------------

for epoch in range(EPOCHS):

    train_loss, train_acc = train_epoch(train_loader)

    val_loss, val_acc = evaluate(val_loader)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")


# -----------------------------
# TEST EVALUATION
# -----------------------------

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs,1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())


print("\nConfusion Matrix")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report")
print(classification_report(all_labels, all_preds, target_names=class_names))

# -----------------------------
# SAVE MODEL
# -----------------------------

torch.save(model.state_dict(), "resnet50_genai_detector.pth")

print("\nModel saved")
