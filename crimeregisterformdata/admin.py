from django.contrib import admin
from .models import CrimeRegisterFormData

# Optional: Create a ModelAdmin class to customize the admin interface
class CrimeRegisterFormDataAdmin(admin.ModelAdmin):
    # list_display = ['date', 'time', 'crime_type', 'location_city']  # columns to display in the list view
    list_display = ['date', 'time', 'crime_type', 'location_city',
                    'latitude','longitude','crime_description',
                    'reported_type','status','injuries',
                    'victims','outcome','news_resources']  # columns to display in the list view
    
    search_fields = ['crime_type', 'location_city', 'reported_type']  # fields to search by
    list_filter = ['crime_type', 'location_city', 'status']  # filters to apply on the right side
    ordering = ['-date', '-time']  # order by date and time in descending order

# Register the model with the admin site
admin.site.register(CrimeRegisterFormData, CrimeRegisterFormDataAdmin)

