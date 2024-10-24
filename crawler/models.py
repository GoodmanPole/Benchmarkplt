from django.db import models

# Create your models here.

class Crypto(models.Model):
    name=models.CharField(max_length=15, blank=False)
    alias=models.CharField(max_length=10, blank=False)
    image=models.ImageField(null=True, blank=True, upload_to="images/")
    status=models.BooleanField(default=False, blank=False)

    # def get_absolute_url(self):
    #     #return f"/product/{self.id}/"
    #
    #     return reverse("products:product", kwargs={"id": self.id})
