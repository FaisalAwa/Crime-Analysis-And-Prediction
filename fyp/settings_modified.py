from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure--24183e-9dmh#oe01ux=%r33ex)r&!1)aa3^9njyjvk2g@x7f8'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = [
    'admin_adminlte.apps.AdminAdminlteConfig',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'cities',
    'csvapp',
    'usersregistration',
    'django_summernote', 
    'statisticalanalysis',
    'crimeregisterformdata',
    'citiesdata'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'fyp.urls'

TEMPLATES = [
    {
        'OPTIONS': {
            'context_processors': [
                # ...
            'fyp.context_processors.user_count_context'# Add this line
            ],
        },
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'fyp.context_processors.user_count_context',
            ],
        },
    },
]

WSGI_APPLICATION = 'fyp.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [ ... ]  # I'm assuming this is unchanged

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'  # added leading slash
STATIC_ROOT = os.path.join(BASE_DIR, 'collected_static')

# Ensure static_files directory exists
if not os.path.exists(os.path.join(BASE_DIR, 'static_files')):
    os.makedirs(os.path.join(BASE_DIR, 'static_files'))

STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static_files'),
]

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media/')

X_FRAME_OPTIONS = 'SAMEORIGIN'

SUMMERNOTE_THEME = 'bs4'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
