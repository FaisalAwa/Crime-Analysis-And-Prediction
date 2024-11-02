from django.db import models
from tinymce.models import HTMLField
class Cities(models.Model):
    city_name = models.CharField(max_length=100)
    city_description =HTMLField()
    image = models.ImageField(upload_to='city_images/')
