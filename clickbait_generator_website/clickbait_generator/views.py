from django.shortcuts import render
from django.http import HttpResponse
from py_model.clickbait_generator import clickbait_generator
from random import randint


# Create your views here.

def home(request):

    if request.method == 'POST':
        clickbait_message = clickbait_generator(10, randint(1, 10000000))
        return render(request, 'clickbait_generator/home.html', {'clickbait_message': clickbait_message})
    else:
        return render(request, 'clickbait_generator/home.html')