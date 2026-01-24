from django.urls import path
from . import views

app_name = 'api'

urlpatterns = [
    path('chart-data/', views.chart_data_api, name='chart_data'),
]
