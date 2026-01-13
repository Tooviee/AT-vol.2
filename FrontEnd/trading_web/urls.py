"""
URL configuration for trading_web project.
"""

from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from trading_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Authentication
    path('login/', auth_views.LoginView.as_view(template_name='trading_app/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    
    # Trading app
    path('', include('trading_app.urls')),
]
