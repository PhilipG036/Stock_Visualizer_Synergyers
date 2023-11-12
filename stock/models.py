from django.db import models
from django.contrib.auth import get_user_model

UserModel = get_user_model()

class UserActivity(models.Model):
    user = models.ForeignKey(to=UserModel, on_delete=models.CASCADE)
    forecast_days = models.IntegerField()
    stock_symbol = models.CharField(max_length=100)
    model_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True, null=True, blank=True)

