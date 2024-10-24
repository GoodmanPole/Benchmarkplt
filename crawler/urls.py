from django.urls import path
from django.contrib.auth import views as auth_views

from . import views

# URL Path
app_name='crawler'
urlpatterns = [


    # Django Auth
    path('', views.crawler_view, name= 'crawler'),
]