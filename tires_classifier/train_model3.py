import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

data_dir = "/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/dataset"
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Для RGB можно: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class_names = dataset.classes
print("Классы:", class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)

# Заменяем последний слой (на 2 класса)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Обучим несколько эпох
epochs = NUM_EPOCHS
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"[{epoch+1}/{epochs}] Loss: {running_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), 'model_finetuned.pth')

# Оценка
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\nОтчёт по метрикам на валидации:")
print(classification_report(y_true, y_pred, target_names=val_dataset.classes))
