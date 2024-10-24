from django.urls import path
from django.contrib.auth import views as auth_views

from . import views

# URL Path
app_name='predictmodels'
urlpatterns = [


    # Django Auth
    path('', views.models_view, name= 'predictmodels'),
]