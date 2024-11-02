# ***************************************

from django.contrib import admin
from django.contrib.auth.models import User
from django_summernote.admin import SummernoteModelAdmin

import csv
from csvapp.models import CSVData
from django.http import HttpResponse

from statisticalanalysis.models import StatisticalAnalysis
from crimeregisterformdata.models import CrimeRegisterFormData

from cities.models import Cities

from contact_form.models import ContactFormEntry




from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

# Optional: Create a ModelAdmin class to customize the User model admin interface
class UserAdmin(BaseUserAdmin):
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff')
    search_fields = ('username', 'email')
    ordering = ('username',)
    filter_horizontal = ('groups', 'user_permissions',)

# Define the export_to_csv action
def export_to_csv(modeladmin, request, queryset):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="crimeregisterformdata.csv"'
    writer = csv.writer(response)
    # write the headers
    field_names = [field.name for field in modeladmin.model._meta.fields]
    writer.writerow(field_names)
    # write the data
    for obj in queryset:
        writer.writerow([getattr(obj, field) for field in field_names])
    return response

export_to_csv.short_description = 'Export to CSV'


class MyAdminSite(admin.AdminSite):
    def index(self, request, extra_context=None):
        extra_context = extra_context or {}

        # Calculate the total number of registered users
        user_count = User.objects.count()
        cities_count = Cities.objects.count()
        analysis_files_uploaded = StatisticalAnalysis.objects.count()
        crime_reported_data = CrimeRegisterFormData.objects.count()

        # Add the user count to the extra_context
        extra_context['user_count'] = user_count
        extra_context['cities_count'] = cities_count
        extra_context['analysis_files_uploaded'] = analysis_files_uploaded
        extra_context['crime_reported_data'] = crime_reported_data

        return super().index(request, extra_context=extra_context)

custom_admin_site = MyAdminSite()

admin.site = custom_admin_site

custom_admin_site.register(User, UserAdmin)

# class CityAdmin(SummernoteModelAdmin):
#     summernote_fields = ('city_description',)

# class CitiesAdmin(SummernoteModelAdmin):
#     summernote_fields = ('description',)

custom_admin_site.register(Cities)

custom_admin_site.register(ContactFormEntry)

custom_admin_site.register(CSVData)

custom_admin_site.register(StatisticalAnalysis)

# Optional: Create a ModelAdmin class to customize the admin interface
class CrimeRegisterFormDataAdmin(admin.ModelAdmin):
    # list_display = ['date', 'time', 'crime_type', 'location_city']  # columns to display in the list view
    list_display = ['date', 'time', 'crime_type', 'location_city',
                    'latitude', 'longitude', 'crime_description',
                    'reported_type', 'status', 'injuries',
                    'victims', 'outcome', 'news_resources']  # columns to display in the list view
    
    search_fields = ['crime_type', 'location_city', 'reported_type']  # fields to search by
    list_filter = ['crime_type', 'location_city', 'status']  # filters to apply on the right side
    ordering = ['-date', '-time']  # order by date and time in descending order
    actions = [export_to_csv]

# Register the model with the admin site
custom_admin_site.register(CrimeRegisterFormData, CrimeRegisterFormDataAdmin)

# i made this section my new admin 


###########################


