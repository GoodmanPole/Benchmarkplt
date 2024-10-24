from django.urls import path
from django.contrib.auth import views as auth_views

from . import views

# URL Path
app_name='extendPortal'
urlpatterns = [


    # Django Auth
    path('', views.editor_view, name='extend'),
    # path('runcode', views.runcode, name='runcode'),
]