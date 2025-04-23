# evaluate_model.py
import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/ai_workshop_project/tires_classifier/model/resnet18_defect_classifier.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Подготовка тестовых данных
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(root="/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/dataset", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Предсказания
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Отчет
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
