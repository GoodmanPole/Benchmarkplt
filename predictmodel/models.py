from django.db import models

# Create your models here.

class PredictModels(models.Model):
    name=models.CharField(max_length=15, blank=False)
    status=models.BooleanField(default=False, blank=False)

    # def get_absolute_url(self):
    #     #return f"/product/{self.id}/"
    #
    #     return reverse("products:product", kwargs={"id": self.id})
