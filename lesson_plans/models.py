from django.db import models
from django.contrib.auth.models import User

class Document(models.Model):
    file = models.FileField(upload_to='documents/', null=True, blank=True)  # Allow null values
    vector = models.JSONField(default=list)  # Store vectors as a list
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class UserProfile(models.Model):
    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('client', 'Client'),
    )
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)

    def __str__(self):
        return f"{self.user.username} - {self.role}"

