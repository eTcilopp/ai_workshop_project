from django.db import models

class TireImage(models.Model):
    image = models.ImageField(upload_to='tire_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    result = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"Tire image uploaded at {self.uploaded_at}"
