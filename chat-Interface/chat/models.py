from django.db import models
from django.contrib.auth.models import User
from django.db import models

class Country(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Province(models.Model):
    name = models.CharField(max_length=100)
    country = models.ForeignKey(Country, on_delete=models.CASCADE, related_name="provinces")

    def __str__(self):
        return f"{self.name}, {self.country.name}"

class City(models.Model):
    name = models.CharField(max_length=100)
    province = models.ForeignKey(Province, on_delete=models.CASCADE, related_name="cities")

    def __str__(self):
        return f"{self.name}, {self.province.name}"

class Property(models.Model):
    PROPERTY_TYPE_CHOICES = [
        ('house', 'Casa'),
        ('cabin', 'Cabaña'),
        ('apartment', 'Apartamento'),
        ('studio', 'Estudio'),
        ('land', 'Terreno'),
    ]
    PROJECT_TYPE_CHOICES = [
        ('pozo', 'Pozo'),
        ('no_pozo', 'No Pozo'),
    ]
    RESIDENCE_TYPE_CHOICES = [
        ('familiar', 'Familiar'),
        ('unipersonal', 'Unipersonal'),
        ('universitaria', 'Universitaria'),
    ]
    PROJECT_CATEGORY_CHOICES = [
        ('social', 'Interés Social'),
        ('luxury', 'Luxury'),
        ('premium', 'Premium'),
    ]

    country = models.ForeignKey(Country, on_delete=models.SET_NULL, null=True, blank=True, related_name="properties")
    province = models.ForeignKey(Province, on_delete=models.SET_NULL, null=True, blank=True, related_name="properties")
    city = models.ForeignKey(City, on_delete=models.SET_NULL, null=True, blank=True, related_name="properties")
    location = models.CharField(max_length=255)  # Ubicación específica (Norte, Sur, etc.)
    price = models.DecimalField(max_digits=10, decimal_places=2)  # Precio del inmueble
    square_meters = models.DecimalField(max_digits=6, decimal_places=2)  # Metros cuadrados del inmueble
    property_type = models.CharField(max_length=20, choices=PROPERTY_TYPE_CHOICES)  # Tipo de inmueble
    project_type = models.CharField(max_length=20, choices=PROJECT_TYPE_CHOICES)  # Tipo de proyecto
    num_rooms = models.IntegerField()  # Número de ambientes
    num_bedrooms = models.IntegerField()  # Número de habitaciones
    residence_type = models.CharField(max_length=20, choices=RESIDENCE_TYPE_CHOICES)  # Tipo de residencia
    project_category = models.CharField(max_length=20, choices=PROJECT_CATEGORY_CHOICES)  # Tipo de proyecto
    description = models.TextField(blank=True, null=True)  # Descripción opcional del inmueble
    image = models.ImageField(upload_to='property_images/', null=True, blank=True)  # Campo de imagen para la propiedad
    url = models.CharField(max_length=255, null=True) # Direccion web del proyecto

    created_at = models.DateTimeField(auto_now_add=True)  # Fecha de creación

    def __str__(self):
        return f"{self.property_type.capitalize()} en {self.city} - {self.price} USD"

class Finanzas(models.Model):
    PLANES_FINANCIAMIENTO_CHOICES = [
        ('hab', 'Habitacional'),
        ('alq', 'Alquiler'),
        ('inv_conjunta', 'Inversión Conjunta'),
    ]

    TIPO_MONEDA_CHOICES = [
        ('usd', 'Dólar'),
        ('eur', 'Euro'),
        ('crypto', 'Cripto'),
    ]

    planes_de_financiamiento = models.CharField(
        max_length=20,
        choices=PLANES_FINANCIAMIENTO_CHOICES,
        verbose_name="Planes de financiamiento"
    )
    tipo_de_inversion = models.CharField(
        max_length=20,
        choices=PLANES_FINANCIAMIENTO_CHOICES,
        verbose_name="Tipo de inversión"
    )
    tipo_de_moneda = models.CharField(
        max_length=10,
        choices=TIPO_MONEDA_CHOICES,
        verbose_name="Tipo de moneda"
    )

    def __str__(self):
        return f"{self.planes_de_financiamiento} - {self.tipo_de_inversion} - {self.tipo_de_moneda}"
# Modelo para el historial de conversaciones
class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    input = models.TextField()
    output = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

class Chunk(models.Model):
    document_id = models.IntegerField()
    content = models.TextField()
    embedding = models.JSONField()

    def __str__(self):
        return f"Document {self.document_id}: {self.content[:50]}"

