# Generated by Django 4.2.2 on 2023-08-17 07:04

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='StatisticalAnalysis',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='analysisfiles/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('csv_name', models.CharField(default='', max_length=255)),
            ],
        ),
    ]
