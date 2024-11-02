"""
URL configuration for fyp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include,path
from fyp import views
# urls.py
from fyp.custom_admin import custom_admin_site
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

from django.urls import path
from . import views
# from usersregistration import views




# ... your url patterns ...


urlpatterns = [
    path('admin/', custom_admin_site.urls), # this is our new and default admin, just one link , okkkkkk.....
    path('admin_dashboard', include('admin_adminlte.urls')),
    path('summernote/', include('django_summernote.urls')),
    path('cardchecking/',views.cardchecking),
    path('signup',views.SignupPage,name='signup'),
    path('contact_form_submission/', views.contact_form_submission, name='contact_form_submission'),
    path('',views.myhomepage, name='myhomepage'),
    path('signin/',views.LoginPage,name='login'),
    path('home/',views.HomePage,name='home'),
    path('logout/',views.LogoutPage,name='logout'),
    path('header/',views.header),
    path('carousel/',views.carousel),
    path('registercrime/',views.registercrime),
    path('footer/',views.footer),
    path('flipcard/',views.flipcard),
    path('hovercards/',views.citiesdata),
    path('homepage/',views.homepage),
    path('iconsmarquee/',views.iconsmarquee),
    path('hometemplate/',views.hometemplate),
    path ('contactform/',views.register_crime,name='contactform'),
    path('karachi/', views.karachi, name='karachi'),
    path('lahore/', views.lahore, name='lahore'),
    path('multan/', views.multan, name='multan'),
    path('faisalabad/', views.faisalabad, name='faisalabad'),
    path('quetta/', views.quetta, name='quetta'),
    path('gujranwala/', views.gujranwala, name='gujranwala'),
    path('peshawar/', views.peshawar, name='peshawar'),
    path('islamabad/', views.islamabad, name='islamabad'),
    path('rawalpindi/', views.rawalpindi, name='rawalpindi'),

    path('pakistan_drone_attacks/',views.pakistan_drone_attacks),
    path('Suicide_bombing_attacks/',views.Suicide_bombing_attacks),
    path('stats/',views.crimestats),
    path('register_crime/',views.register_crime,name="register_crime"),
    path('angularjs/',views.angularjs, name='angularjs'),
    path('provinces/',views.provincewisedata),
    path('secondhome',views.secondtemplate),
    path('statistics_page',views.statistics_page),
    path('usersearch',views.searchingarea),

    path('operation_al_mizan/',views.operation_al_mizan),
    path('operation_rah_e_haq/',views.operation_rah_e_haq),
    path('operation_rah_e_nijaat/',views.operation_rah_e_nijaat),
    path('operation_sherdil/',views.operation_sherdil),
    path('operation_zarb_e_azb/',views.operation_zarb_e_azb),
    path('operation_black_thunderstorm/',views.operation_black_thunderstorm),
    path('operation_raddul_fasaad/',views.operation_raddul_fasaad),
    path('karachi_operation/',views.karachi_operation)


]

from django.conf import settings
from django.conf.urls.static import static

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)





