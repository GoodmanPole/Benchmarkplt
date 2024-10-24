from django.shortcuts import render, get_object_or_404

def home_view(request,*args,**kwargs):
    # return HttpResponse("<h1>Hello Goodman</h1>")
    return render(request,"index.html",{})

def about_view(request,*args,**kwargs):
    # return HttpResponse("<h1>Hello Goodman</h1>")
    return render(request,"about.html",{})







