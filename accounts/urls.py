from django.urls import path
from django.contrib.auth import views as auth_views

# URL Path

urlpatterns = [


    # Django Auth
    path('', auth_views.LoginView.as_view(template_name="accounts/login.html"), name= 'login'),
]