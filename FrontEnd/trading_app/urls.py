"""
URL configuration for trading_app
"""

from django.urls import path
from . import views

app_name = 'trading_app'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('trading-log/', views.trading_log, name='trading_log'),
    path('positions/', views.positions, name='positions'),
    path('performance/', views.performance, name='performance'),
    path('risk/', views.risk_dashboard, name='risk_dashboard'),
    path('market-status/', views.market_status, name='market_status'),
    path('ml-insights/', views.ml_insights, name='ml_insights'),
]
