# Generated by Django 3.2.12 on 2022-04-09 21:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('crawler', '0002_rename_title_crypto_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='crypto',
            name='image',
            field=models.ImageField(blank=True, null=True, upload_to=''),
        ),
    ]