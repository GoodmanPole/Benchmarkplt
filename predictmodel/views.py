from django.shortcuts import render, redirect
from .models import PredictModels

from .PredictionModels import PredictionModels

# Create your views here.

def models_view(request,*args,**kwargs):
    # return HttpResponse("<h1>Hello Goodman</h1>")
    models_list= PredictModels.objects.all()

    if request.method == "POST":
        # Get the name's of the chosen models
        name_list = request.POST.getlist('boxes')

        # Uncheck all Models
        name_list.update(status=False)

        for n in name_list:
            PredictModels.objects.filter(name=n).update(status=True)


        # call the models

        model = PredictionModels()
        for c in name_list:
            print(c)
            method = getattr(PredictionModels, c.lower())
            method(model)

        return redirect('predictmodels:predictmodels')


    else:
        return render(request,"predictmodels.html",{"models_list": models_list})