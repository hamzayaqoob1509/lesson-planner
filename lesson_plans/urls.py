from django.urls import path
from . import views

urlpatterns = [
   path('', views.upload_document, name='upload_document'),
#    path('search/', views.search_lessons, name='search_lessons'),
   path('search/', views.search_view, name='search_view'),

]
