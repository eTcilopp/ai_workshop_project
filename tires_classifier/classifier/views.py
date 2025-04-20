from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from django.core.files.base import ContentFile
from .models import TireImage


def index(request):
    return render(request, 'classifier/index.html')


@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        try:
            image_data = request.POST.get('image')
            if image_data:
                # Remove the data:image/jpeg;base64, prefix
                format, imgstr = image_data.split(';base64,')
                ext = format.split('/')[-1]

                # Create a ContentFile from the base64 data
                data = ContentFile(base64.b64decode(imgstr), name='temp.' + ext)

                # Save the image
                tire_image = TireImage(image=data)
                tire_image.save()

                # Here you would add your tire classification logic
                # For now, we'll return a dummy result
                result = "Example tire classification result"
                tire_image.result = result
                tire_image.save()

                return JsonResponse({
                    'success': True,
                    'result': result
                })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    })
