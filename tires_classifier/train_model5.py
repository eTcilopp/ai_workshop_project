import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


# --- Настройки ---
train_data_dir = "/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/dataset"
val_data_dir = "/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/dataset_split/val"
num_classes = 2
batch_size = 32
num_epochs = 50
learning_rate = 0.001

# Проверка на GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Трансформации ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- Загрузка данных ---

train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_data_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# --- Модель ---
model = models.resnet18(weights="IMAGENET1K_V1")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# --- Оптимизатор и функция потерь ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Для графиков ---
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# --- Обучение ---
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # ---- Train ----
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(train_acc)

    # ---- Validation ----
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {epoch_val_loss:.4f}, Val   Acc: {val_acc:.4f}")

# --- Отчёт ---
print("\nОтчёт по метрикам на валидации:")
target_names = train_dataset.classes
print(classification_report(all_labels, all_preds, target_names=target_names))

# --- Сохранение модели ---
torch.save(model.state_dict(), "model_resnet18_defect_tires3.pt")

# --- Построение графиков ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Val Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("training_curves.png")
print("\nГрафик сохранён как training_curves.png")
