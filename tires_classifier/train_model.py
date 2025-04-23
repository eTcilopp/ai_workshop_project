import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Константы
DATA_DIR = "/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/dataset"  # Убедитесь, что внутри есть папки 'defective' и 'good'
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Преобразования изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Загрузка датасета
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Проверка классов
print("Классы:", dataset.class_to_idx)  # {'defective': 0, 'good': 1}

# Модель
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Классификация на 2 класса
model.to(DEVICE)

# Функция потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Обучение модели
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

# Сохранение модели
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/resnet18_defect_classifier.pth")
print("✅ Модель сохранена в: model/resnet18_defect_classifier.pth")
