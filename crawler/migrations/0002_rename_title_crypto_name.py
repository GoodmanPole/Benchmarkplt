# Generated by Django 3.2.12 on 2022-04-09 21:25

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('crawler', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='crypto',
            old_name='title',
            new_name='name',
        ),
    ]
