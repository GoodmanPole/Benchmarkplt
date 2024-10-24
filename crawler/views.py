from django.shortcuts import render, redirect
from subprocess import run, PIPE
import sys
from .CoinMarketCapCrawler import CoinMarketCapCrawler

from .models import Crypto



# Create your views here.

def crawler_view(request,*args,**kwargs):
    # return HttpResponse("<h1>Hello Goodman</h1>")
    crypto_list= Crypto.objects.all()

    if request.method == "POST":
        # Get Chosen Crypto's Name
        name_list= request.POST.getlist('boxes')
        print(name_list)

        # Uncheck all Crypto
        crypto_list.update(status=False)

        for n in name_list:
            Crypto.objects.filter(name=n).update(status=True)

        starty= request.POST.get('startyear')
        endy= request.POST.get('endyear')
        print(starty, endy)

        # Call the crawler
        crawl=CoinMarketCapCrawler(name_list,starty,endy)
        crawl.crawler()






        return redirect('crawler:crawler')
    else:
        return render(request,"crawler.html",{"crypto_list": crypto_list})
