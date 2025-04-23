import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

# Проверка устройства
device = torch.device("cpu")

# Класс датасета
class SimpleBoxDataset(Dataset):
    def __init__(self, image_dir, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.images = sorted(os.listdir(image_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transforms:
            img = self.transforms(img)

        boxes = torch.tensor([[50, 50, 200, 200]], dtype=torch.float32)  # фиксированный бокс
        labels = torch.tensor([1], dtype=torch.int64)  # один класс

        target = {"boxes": boxes, "labels": labels}
        return img, target

    def __len__(self):
        return len(self.images)

# Трансформации
transform = transforms.Compose([
    transforms.ToTensor()
])

# Пути
image_dir = "/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/dataset/defective"

# Загрузка данных
dataset = SimpleBoxDataset(image_dir=image_dir, transforms=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Модель
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
model.to(device)

# Оптимайзер
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-4)

# Обучение
print("=== Начало обучения ===")
model.train()
for epoch in range(3):
    running_loss = 0.0
    for images, targets in loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}], Loss: {running_loss:.4f}")

# Сохранение модели
torch.save(model.state_dict(), "simple_resnet_50_test.pth")
print("Модель сохранена.")

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
    drawn = draw_bounding_boxes(
         (img_tensor[0] * 255).byte().cpu(), 
         boxes[selected], 
         labels=labels[selected], 
         colors="red"
    )
    plt.imshow(drawn.permute(1, 2, 0))
    plt.title("Predictions")
    plt.axis("off")
    plt.show()

plt.savefig("/mnt/c/Users/ecopa/Desktop/Proekts/Шины/Work/ai_workshop_project/tires_classifier/training_curves.png")
print("\nГрафик сохранён как training_curves.png")
