from django.db import models

class CSVData(models.Model):
    file = models.FileField(upload_to='csvfiles/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    csv_name = models.CharField(max_length=255, default='')


    def __str__(self):
        return self.csv_name
