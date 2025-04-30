import requests
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from .models import Chunk, Conversation, Property
import json
import logging
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import re
from django.conf import settings
from django.http import HttpResponse
import os

# Configuración de logging
logger = logging.getLogger(__name__)

# Variables de configuración
API_BASE_URL = getattr(settings, 'API_BASE_URL', 'http://fastapi:8000')
API_TIMEOUT = getattr(settings, 'API_TIMEOUT', 30)

@csrf_exempt
def get_all_data(request):
    """
    Endpoint para obtener todos los datos de Conversation, Chunk y Property.
    """
    if request.method == 'GET':
        # Obtener datos de los modelos y excluir 'timestamp' y 'created_at'
        conversations = list(Conversation.objects.all().values('input', 'output'))
        chunks = list(Chunk.objects.all().values('document_id', 'content', 'embedding'))
        properties = list(Property.objects.all().values('location', 'price', 'square_meters', 'property_type', 'description'))

        # Formatear la respuesta JSON
        data = {
            "conversations": conversations,
            "chunks": chunks,
            "properties": properties,
        }

        # Enviar los datos como respuesta
        return JsonResponse(data, safe=False)
    else:
        return JsonResponse({"error": "Only GET requests are allowed."}, status=400)

@csrf_exempt
def send_data_to_fastapi(request):
    """
    Envía datos desde la base de datos a un servicio FastAPI externo.
    """
    # Función para eliminar campos datetime de los datos
    def exclude_datetime_fields(data):
        for item in data:
            item.pop('timestamp', None)  # Elimina el campo 'timestamp' si existe
            item.pop('created_at', None)  # Elimina el campo 'created_at' si existe
        return data

    # Recopilar datos de los modelos
    conversations = list(Conversation.objects.all().values())
    chunks = list(Chunk.objects.all().values())
    properties = list(Property.objects.all().values())

    # Eliminar campos datetime de los datos
    conversations = exclude_datetime_fields(conversations)
    properties = exclude_datetime_fields(properties)

    data = {
        "conversations": conversations,
        "chunks": chunks,  # No tiene campos datetime, así que lo dejamos como está
        "properties": properties,
    }

    # Enviar datos a FastAPI
    fastapi_url = "http://fastapi:8000/generate_pdf/"  # Usar nombre de servicio en lugar de IP
    try:
        response = requests.post(fastapi_url, json=data, timeout=API_TIMEOUT)
        
        if response.status_code == 200:
            return JsonResponse({"message": "Data sent successfully", "response": response.json()})
        else:
            logger.error(f"Error sending data to FastAPI: {response.status_code} - {response.text}")
            return JsonResponse({"error": f"Failed to send data: {response.status_code}"}, status=response.status_code)
    except requests.RequestException as e:
        logger.error(f"Request exception when sending data to FastAPI: {str(e)}")
        return JsonResponse({"error": f"Connection error: {str(e)}"}, status=500)

@login_required
def api_chat(request):
    """
    Procesa las solicitudes de chat del usuario y las envía al servicio FastAPI.
    """
    if request.method == 'POST':
        try:
            # Leer el JSON enviado desde el frontend
            body_unicode = request.body.decode('utf-8')
            body = json.loads(body_unicode)
            mensaje = body.get('mensaje')

            # Verificar si el mensaje se recibió correctamente
            logger.info(f"Mensaje recibido: {mensaje}")

            # Usar nombre de servicio Docker en lugar de IP hardcodeada
            api_url = f"{API_BASE_URL}/query"
            
            # Formato correcto para el endpoint /query según main.py
            data = {
                "query": mensaje,
                "conversation_id": None
            }
            
            logger.info(f"Enviando solicitud a: {api_url} con datos: {data}")
            
            # Realizar la petición a la API de FastAPI con timeout
            response = requests.post(
                api_url,
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=API_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                respuesta = data.get('response', '')
                
                # Guardar la conversación
                conversation = Conversation.objects.create(
                    user=request.user,
                    input=mensaje,
                    output=respuesta
                )
                
                # Determinar si formatear como propiedad o chat normal
                if any(keyword in respuesta.lower() for keyword in ['propiedad', 'apartamento', 'casa', 'precio']):
                    formatted_response = format_property_text_response(respuesta)
                else:
                    formatted_response = format_chat_response(respuesta)
                
                return JsonResponse({
                    'type': 'html',
                    'mensaje': formatted_response
                })
            else:
                logger.error(f"Error en la API de FastAPI: Status {response.status_code}, Respuesta: {response.text}")
                return JsonResponse({
                    'type': 'error',
                    'mensaje': f'Error del servidor: {response.status_code}'
                }, status=500)

        except requests.exceptions.Timeout:
            logger.error("Timeout al conectar con la API de FastAPI")
            return JsonResponse({
                'type': 'error',
                'mensaje': 'El servidor está tardando demasiado en responder. Por favor, intenta más tarde.'
            }, status=504)
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Error de conexión con la API de FastAPI: {str(e)}")
            return JsonResponse({
                'type': 'error',
                'mensaje': 'No se pudo establecer conexión con el servicio de chat.'
            }, status=502)
        except json.JSONDecodeError as e:
            logger.error(f"Error decodificando JSON: {str(e)}")
            return JsonResponse({
                'type': 'error',
                'mensaje': 'Error en el formato de la solicitud'
            }, status=400)
        except Exception as e:
            logger.error(f"Error general: {str(e)}", exc_info=True)
            return JsonResponse({
                'type': 'error',
                'mensaje': 'Error interno del servidor'
            }, status=500)

    return JsonResponse({
        'type': 'error',
        'mensaje': 'Método no permitido'
    }, status=405)

def format_property_text_response(text):
    """
    Formatea el texto que contiene información de propiedades en HTML estructurado
    """
    # Dividir el texto en secciones si contiene múltiples propiedades
    properties = text.split('\n\n')
    
    html_response = '<div class="chat-properties">'
    
    for prop in properties:
        if prop.strip():
            # Extraer información clave usando patrones comunes
            location_match = re.search(r'(?:en|ubicado en)\s+([^\.]+)', prop, re.IGNORECASE)
            price_match = re.search(r'(?:USD|precio[:]?\s+)\s*([\d,\.]+)', prop, re.IGNORECASE)
            area_match = re.search(r'(\d+)\s*m²', prop)
            
            html_response += '<div class="property-card">'
            
            # Título/Ubicación
            if location_match:
                html_response += f'<h3 class="property-location">{location_match.group(1)}</h3>'
            
            # Detalles principales
            html_response += '<div class="property-details">'
            if price_match:
                try:
                    price = float(price_match.group(1).replace(',', '').replace('.', ''))
                    html_response += f'<p class="property-price">USD {price:,.2f}</p>'
                except ValueError:
                    logger.error(f"Error convirtiendo precio: {price_match.group(1)}")
                    html_response += f'<p class="property-price">USD {price_match.group(1)}</p>'
            if area_match:
                html_response += f'<p class="property-area">{area_match.group(1)} m²</p>'
            
            # Descripción
            html_response += f'<p class="property-description">{prop}</p>'
            html_response += '</div></div>'
    
    html_response += '</div>'
    return html_response

def format_chat_response(text):
    """
    Formatea respuestas de chat normales en HTML
    """
    # Convertir URLs en enlaces clickeables
    text = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank">\1</a>', text)
    
    # Convertir saltos de línea en <br>
    text = text.replace('\n', '<br>')
    
    return f'<div class="chat-message">{text}</div>'
    
def index(request):
    """
    Vista principal que renderiza la página inicial de la aplicación
    """
    return render(request, 'index.html')

@csrf_protect
def api_view(request):
    """
    API view alternativa (mejorada para usar los mismos estándares que api_chat)
    """
    if request.method == "POST":
        user_input = request.POST.get('mensaje', '')
        try:
            # Usar la misma URL base que api_chat
            api_url = f"{API_BASE_URL}/query"
            
            response = requests.post(
                api_url,
                json={'query': user_input},
                headers={'Content-Type': 'application/json'},
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return JsonResponse({'mensaje': response_data.get('response', '')})
            else:
                logger.error(f'Error de estado desde FastAPI: {response.status_code} - {response.text}')
                return JsonResponse({'error': 'Error con el servicio de chat'}, status=response.status_code)
        except requests.exceptions.RequestException as e:
            logger.error(f'Excepción al conectar con FastAPI: {str(e)}')
            return JsonResponse({'error': 'Error de conexión con el servicio de chat'}, status=500)
    else:
        return JsonResponse({'error': 'Método no permitido'}, status=405)

@csrf_protect
def register(request):
    """
    Maneja el registro de nuevos usuarios
    """
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        email = request.POST['email']

        if User.objects.filter(username=username).exists():
            messages.error(request, 'El nombre de usuario ya existe.')
            return redirect('register')
        else:
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()
            messages.success(request, 'Usuario registrado exitosamente.')
            return redirect('login')
    return render(request, 'register.html')

@csrf_protect
def user_login(request):
    """
    Maneja el inicio de sesión de usuarios
    """
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, 'Inicio de sesión exitoso.')
            return redirect('index')
        else:
            messages.error(request, 'Usuario o contraseña incorrecta.')
            return redirect('login')
    return render(request, 'login.html')

def user_logout(request):
    """
    Maneja el cierre de sesión de usuarios
    """
    logout(request)
    messages.success(request, 'Sesión cerrada exitosamente.')
    return redirect('login')

@csrf_exempt
def save_vectorization(request):
    """
    Endpoint para guardar datos de vectorización en la base de datos
    """
    if request.method == 'POST':
        try:
            logger.info(f"Request body: {request.body}")
            data = json.loads(request.body)
            logger.info(f"Data received: {data}")
            
            # Crear registros en bulk para mayor eficiencia
            chunks_to_create = [
                Chunk(
                    document_id=item['document_id'],
                    content=item['content'],
                    embedding=item['embedding']
                ) for item in data
            ]
            
            Chunk.objects.bulk_create(chunks_to_create)
            
            return JsonResponse({
                "status": "success",
                "message": f"Guardados {len(chunks_to_create)} chunks exitosamente"
            })
        except Exception as e:
            logger.error(f"Error processing data: {e}", exc_info=True)
            return JsonResponse({"status": "fail", "error": str(e)}, status=400)
    return JsonResponse({"status": "fail", "message": "Método no permitido"}, status=400)

def debug_media(request):
    """
    Vista de debug para verificar la configuración de media
    """
    media_root = settings.MEDIA_ROOT
    try:
        media_files = os.listdir(media_root)
        response = f"MEDIA_ROOT: {media_root}\n\nArchivos encontrados:\n"
        for file in media_files:
            file_path = os.path.join(media_root, file)
            size = os.path.getsize(file_path)
            response += f"\n- {file} ({size} bytes)"
        return HttpResponse(response, content_type='text/plain')
    except Exception as e:
        return HttpResponse(f"Error: {str(e)}\nMEDIA_ROOT: {media_root}", content_type='text/plain')

def test_media(request, filename):
    """
    Prueba de acceso a archivos media
    """
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return HttpResponse(f.read(), content_type='image/jpeg')
        else:
            return HttpResponse(f"File not found: {file_path}", status=404)
    except Exception as e:
        return HttpResponse(f"Error: {str(e)}", status=500)
    
def health_check(request):
    """
    Endpoint de verificación de salud para monitoreo
    """
    return JsonResponse({"status": "healthy"})