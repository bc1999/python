"""
URL configuration for decoupled_dj project.

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
# from django.urls import path

# urlpatterns = [
#     path('admin/', admin.site.urls),
# ]

from django.urls import path, include

from django.conf import settings # added new

urlpatterns = [
        path("billing/", include("billing.urls", namespace="billing")),
        path("auth/", include("login.urls", namespace="auth"))
    
    ]

# -------------------
# added ramdomness to admin url to mitigate automated brute force attacks

if settings.DEBUG:
    urlpatterns = [
        path("admin/", admin.site.urls),
    ] + urlpatterns

if not settings.DEBUG:
    urlpatterns = [
        path("77randomAdmin@33/", admin.site.urls),
    ] + urlpatterns

# -------------------
