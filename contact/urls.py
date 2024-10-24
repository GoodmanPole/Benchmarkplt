from django.urls import path
from django.contrib.auth import views as auth_views

from . import views

# URL Path
app_name='contact'
urlpatterns = [


    # Django Auth
    path('', views.contact_view, name= 'contact'),
]