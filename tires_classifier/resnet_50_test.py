import os
import time
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.datasets import VOCDetection
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Проверка на GPU
device = torch.device("cpu")

# Настройки
NUM_CLASSES = 2  # background и 1 класс
EPOCHS = 5
BATCH_SIZE = 1

# Пути к данным
train_data_dir = "/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/dataset"
val_data_dir = "/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/dataset_split/val"


# Кастомный датасет
class MyDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.images = sorted(os.listdir(image_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        #ann_path = os.path.join(self.annotation_dir, os.path.splitext(self.images[idx])[0] + ".txt")
        img = Image.open(img_path).convert("RGB")

        # Здесь пример дефолтной аннотации, замените своей логикой
        boxes = torch.tensor([[50, 50, 200, 200]], dtype=torch.float32)  # xmin, ymin, xmax, ymax
        labels = torch.tensor([1], dtype=torch.int64)  # один класс, например "дефект"

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)

# Преобразования
transform = transforms.Compose([
    transforms.ToTensor()
])

# Датасеты и загрузчики
train_dataset = MyDetectionDataset(
    image_dir="/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/dataset/defective",
    annotation_dir="/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/dataset/good",
    transforms=transform
)
val_dataset = MyDetectionDataset(
    image_dir="/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/dataset_split/train",
    annotation_dir="/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/dataset_split/val",
    transforms=transform
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Модель
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(device)

# Оптимизатор
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-4)

# Обучение
print("=== Начало обучения ===")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for images, targets in train_loader:
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            num_batches += 1
        except Exception as e:
            print(f"[!] Ошибка в батче {num_batches + 1}: {e}")
            continue

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
    print(f"[Epoch {epoch + 1}] Средняя потеря: {avg_loss:.4f}")
    
# Сохранение
torch.save(model.state_dict(), "model_resnet50_detect_tires.pth")
print("Модель сохранена.")

# Оценка
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for images, targets in val_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for output, target in zip(outputs, targets):
            gt_label = target["labels"].cpu().numpy()[0]
            pred_scores = output["scores"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            if len(pred_scores) > 0 and pred_scores[0] >= 0.5:
                y_pred.append(pred_labels[0])
            else:
                y_pred.append(0)
            y_true.append(gt_label)

print("\n=== Отчёт по метрикам ===")
print(classification_report(y_true, y_pred, target_names=["background", "object"]))

# Визуализация
def visualize(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)[0]
    boxes = output["boxes"].cpu()
    labels = output["labels"].cpu()
    scores = output["scores"].cpu()

    selected = scores > 0.5
    drawn = draw_bounding_boxes((img_tensor[0] * 255).byte().cpu(), boxes[selected], labels=labels[selected], colors="red")
    plt.imshow(drawn.permute(1, 2, 0))
    plt.title("Predictions")
    plt.axis("off")
    plt.show()

plt.savefig("/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/ai_workshop_project/tires_classifier/training_curves.png")
print("\nГрафик сохранён как training_curves.png")
