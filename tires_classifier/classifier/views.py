from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import TireImage
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import os
import io
import base64
from django.core.files.base import ContentFile


def index(request):
    return render(request, 'classifier/index.html')


@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        try:
            # 1. Получение изображения (файл или base64)
            if 'file' in request.FILES:
                image_file = request.FILES['file']
                tire_image = TireImage(image=image_file)
                tire_image.save()
                image = Image.open(image_file).convert('RGB')
            else:
                image_data = request.POST.get('image')
                if not image_data:
                    raise ValueError("No image data provided")

                format, imgstr = image_data.split(';base64,')
                ext = format.split('/')[-1]
                data = ContentFile(base64.b64decode(imgstr), name=f'temp.{ext}')

                tire_image = TireImage(image=data)
                tire_image.save()
                image = Image.open(io.BytesIO(base64.b64decode(imgstr))).convert('RGB')

            # 2. Преобразование изображения (как при обучении)
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image_tensor = preprocess(image).unsqueeze(0)

            # 3. Загрузка модели (совместимо с обучением)
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models/resnet18', 'best_tire_classifier.pth')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Создаем модель так же, как при обучении
            model = models.resnet18(weights=None)
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, 2)
            model = model.to(device)

            # Загружаем сохраненные веса
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # 4. Получение предсказания
            with torch.no_grad():
                output = model(image_tensor.to(device))
                _, predicted = torch.max(output, 1)
                class_idx = predicted.item()

            # 5. Сопоставление класса с названием (как при обучении)
            class_names = ['damaged', 'good']  # Должно соответствовать обучению
            if 'class_to_idx' in checkpoint:
                idx_to_class = {v: k for k, v in checkpoint['class_to_idx'].items()}
                result = idx_to_class.get(class_idx, class_names[class_idx])
            else:
                result = class_names[class_idx]

            # 6. Сохранение и возврат результата
            tire_image.result = result
            tire_image.save()

            return JsonResponse({
                'success': True,
                'result': result,
                'image_url': tire_image.image.url
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)

    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    }, status=405)
