# Generated by Django 3.2.12 on 2022-04-10 07:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('crawler', '0003_crypto_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='crypto',
            name='image',
            field=models.ImageField(blank=True, null=True, upload_to='images/'),
        ),
    ]
