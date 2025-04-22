from django.contrib import admin
from import_export.admin import ImportExportModelAdmin  # Para importar/exportar
from .models import Conversation, Chunk, Property, Country, Province, City, Finanzas  # Asegúrate de importar todos los modelos
# Registrar el modelo Finanzas en el admin
@admin.register(Finanzas)
class FinanzasAdmin(ImportExportModelAdmin):
    list_display = ('planes_de_financiamiento', 'tipo_de_inversion', 'tipo_de_moneda')  # Columnas a mostrar
    search_fields = ('planes_de_financiamiento', 'tipo_de_inversion', 'tipo_de_moneda')  # Campos por los que se puede buscar
    list_filter = ('planes_de_financiamiento', 'tipo_de_inversion', 'tipo_de_moneda')  # Filtros laterales
    ordering = ('planes_de_financiamiento',)  # Ordenar por el plan de financiamiento
# Registrar el modelo Conversation en el admin
@admin.register(Conversation)
class ConversationAdmin(ImportExportModelAdmin):
    list_display = ('user', 'input', 'output', 'timestamp')  # Columnas a mostrar
    search_fields = ('user__username', 'input', 'output')  # Campos por los que se puede buscar
    list_filter = ('user', 'timestamp')  # Filtros laterales
    ordering = ('-timestamp',)  # Ordenar por las conversaciones más recientes

# Registrar el modelo Chunk en el admin
@admin.register(Chunk)
class ChunkAdmin(ImportExportModelAdmin):
    list_display = ('document_id', 'content_summary')  # Mostrar parte del contenido
    search_fields = ('document_id', 'content')  # Permitir búsqueda por ID y contenido
    ordering = ('document_id',)  # Ordenar por el ID del documento
    def content_summary(self, obj):
        return obj.content[:50]  # Mostrar los primeros 50 caracteres del contenido
    content_summary.short_description = 'Content Summary'

# Registrar el modelo Country en el admin
@admin.register(Country)
class CountryAdmin(ImportExportModelAdmin):
    list_display = ('name',)
    search_fields = ('name',)
    ordering = ('name',)

# Registrar el modelo Province en el admin
@admin.register(Province)
class ProvinceAdmin(ImportExportModelAdmin):
    list_display = ('name', 'country')
    search_fields = ('name', 'country__name')
    list_filter = ('country',)
    ordering = ('name',)

# Registrar el modelo City en el admin
@admin.register(City)
class CityAdmin(ImportExportModelAdmin):
    list_display = ('name', 'province')
    search_fields = ('name', 'province__name', 'province__country__name')
    list_filter = ('province',)
    ordering = ('name',)

# Registrar el modelo Property en el admin
@admin.register(Property)
class PropertyAdmin(ImportExportModelAdmin):
    list_display = ('get_country', 'get_province', 'get_city', 'location', 'price', 'square_meters', 'get_property_type', 'get_project_type', 'get_residence_type', 'get_project_category', 'created_at')
    search_fields = ('location', 'property_type', 'country__name', 'province__name', 'city__name')
    list_filter = ('property_type', 'created_at')
    ordering = ('-created_at',)

    # Para mostrar dependencias jerárquicas en el formulario de administración
    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if db_field.name == "province":
            if 'country' in request.GET:
                kwargs["queryset"] = Province.objects.filter(country_id=request.GET['country'])
            else:
                kwargs["queryset"] = Province.objects.none()
        elif db_field.name == "city":
            if 'province' in request.GET:
                kwargs["queryset"] = City.objects.filter(province_id=request.GET['province'])
            else:
                kwargs["queryset"] = City.objects.none()
        return super().formfield_for_foreignkey(db_field, request, **kwargs)

    # Métodos para obtener los valores de los campos relacionados
    def get_country(self, obj):
        return obj.country.name if obj.country else None
    get_country.short_description = 'Country'

    def get_province(self, obj):
        return obj.province.name if obj.province else None
    get_province.short_description = 'Province'

    def get_city(self, obj):
        return obj.city.name if obj.city else None
    get_city.short_description = 'City'

    def get_property_type(self, obj):
        return obj.get_property_type_display()
    get_property_type.short_description = 'Property Type'

    def get_project_type(self, obj):
        return obj.get_project_type_display()
    get_project_type.short_description = 'Project Type'

    def get_residence_type(self, obj):
        return obj.get_residence_type_display()
    get_residence_type.short_description = 'Residence Type'

    def get_project_category(self, obj):
        return obj.get_project_category_display()
    get_project_category.short_description = 'Project Category'
