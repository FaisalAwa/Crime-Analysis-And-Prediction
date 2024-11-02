from django.db import models

# Create your models here.

class CrimeRegisterFormData(models.Model):
    date = models.DateField()
    time = models.TimeField()
    CRIME_CHOICES = [
        ('Theft', 'Theft'),
        ('Assault', 'Assault'),
        ('Robbery', 'Robbery'),
        ('Murder', 'Murder'),
        ('Fraud', 'Fraud'),
        # ... add more choices as needed ...
    ]
    crime_type = models.CharField(max_length=50, choices=CRIME_CHOICES)
    CITY_CHOICES = [
        ('Karachi', 'Karachi'),
        ('Lahore', 'Lahore'),
        ('Islamabad', 'Islamabad'),
        ('Rawalpindi', 'Rawalpindi'),
        ('Faisalabad', 'Faisalabad'),
        # ... add more cities as needed ...
    ]
    location_city = models.CharField(max_length=50, choices=CITY_CHOICES)
    latitude = models.FloatField()
    longitude = models.FloatField()
    crime_description = models.TextField()
    reported_type = models.CharField(max_length=200)
    STATUS_CHOICES = [
        ('alone', 'Alone'),
        ('died', 'Died')
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES)
    injuries = models.PositiveIntegerField()
    victims = models.PositiveIntegerField()
    outcome = models.TextField()
    news_resources = models.TextField()

    def __str__(self):
        return f"{self.date} - {self.crime_type} - {self.location_city}"
