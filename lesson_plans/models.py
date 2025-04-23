from django.db import models

class Document(models.Model):
    file = models.FileField(upload_to='documents/', null=True, blank=True)  # Allow null values
    vector = models.JSONField(default=list)  # Store vectors as a list
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
