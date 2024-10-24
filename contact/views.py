from django.shortcuts import render

# Create your views here.

def contact_view(request,*args,**kwargs):
    # return HttpResponse("<h1>Hello Goodman</h1>")
    return render(request,"contact.html",{})