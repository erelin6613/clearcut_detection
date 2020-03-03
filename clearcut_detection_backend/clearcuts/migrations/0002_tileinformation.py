# Generated by Django 2.2.3 on 2020-02-20 20:36

import django.contrib.gis.db.models.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('clearcuts', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='TileInformation',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tile_name', models.CharField(max_length=5)),
                ('tile_location', models.CharField(blank=True, max_length=60, null=True)),
                ('tile_metadata_hash', models.CharField(blank=True, max_length=32, null=True)),
                ('cloud_coverage', models.FloatField(default=0)),
                ('mapbox_tile_id', models.CharField(blank=True, max_length=32, null=True)),
                ('mapbox_tile_name', models.CharField(blank=True, max_length=32, null=True)),
                ('mapbox_tile_layer', models.CharField(blank=True, max_length=32, null=True)),
                ('coordinates', django.contrib.gis.db.models.fields.PolygonField(blank=True, null=True, srid=4326)),
            ],
        ),
    ]