from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def home(request):
    if request.method == 'POST':
        clickbait_message = "Hi, I'm a clickbait message."
        return render(request, 'clickbait_generator/home.html', {'clickbait_message': clickbait_message})
    else:
        return render(request, 'clickbait_generator/home.html')