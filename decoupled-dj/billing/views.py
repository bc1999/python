# from django.shortcuts import render

# Create your views here.

from django.views.generic import TemplateView

from django.contrib.auth.mixins import LoginRequiredMixin # newly added

# class Index(TemplateView):
class Index(LoginRequiredMixin, TemplateView): # revised
    template_name = "billing/index.html"