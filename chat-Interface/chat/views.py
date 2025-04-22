import requests
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import authenticate, login, logout
from django.views.decorators.csrf import csrf_protect, csrf_exempt  # Cambio de csrf_exempt a csrf_protect
from .models import Chunk  # Asegúrate de importar Chunk
import json
import logging
from django.contrib.auth.models import User  # Agrega esta línea
from django.contrib import messages
logger = logging.getLogger(__name__)
import requests
from django.http import JsonResponse
from .models import Conversation  # Importar el modelo de conversación
from django.contrib.auth.decorators import login_required
import requests
from django.http import JsonResponse
from .models import Conversation, Chunk, Property

from django.http import JsonResponse
from .models import Conversation, Chunk, Property
from django.views.decorators.csrf import csrf_exempt
import re
from django.conf import settings
from django.http import HttpResponse
import os


logger = logging.getLogger(__name__)

@csrf_exempt
def get_all_data(request):
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
    fastapi_url = "http://137.184.19.215:8800/generate_pdf/"  # Cambia si FastAPI está en otra dirección
    response = requests.post(fastapi_url, json=data)

    if response.status_code == 200:
        return JsonResponse({"message": "Data sent successfully", "response": response.json()})
    else:
        return JsonResponse({"error": "Failed to send data"}, status=response.status_code)

# @login_required
# def api_chat(request):
#     if request.method == 'POST':
#         try:
#             # Leer el JSON enviado desde el frontend
#             body_unicode = request.body.decode('utf-8')
#             body = json.loads(body_unicode)
#             mensaje = body.get('mensaje')

#             # Verificar si el mensaje se recibió correctamente
#             logger.info(f"Mensaje recibido: {mensaje}")

#             # Realizar la petición a la API de FastAPI
#             response = requests.post(
#                 'https://api.dnlproptech-chat.com/chat/',
#                 json={'user_input': mensaje},
#                 headers={'Content-Type': 'application/json'}
#             )

#             if response.status_code == 200:
#                 data = response.json()
#                 respuesta = data.get('response', '')

#                 # Guardar la conversación
#                 conversation = Conversation.objects.create(
#                     user=request.user,
#                     input=mensaje,
#                     output=respuesta
#                 )

#                 # Formatear la respuesta HTML según el contenido
#                 if isinstance(respuesta, str):
#                     # Procesar el texto para identificar propiedades
#                     if any(keyword in respuesta.lower() for keyword in ['propiedad', 'apartamento', 'casa', 'precio']):
#                         # Formatear texto como una descripción de propiedad estructurada
#                         formatted_response = format_property_text_response(respuesta)
#                     else:
#                         # Formatear texto normal
#                         formatted_response = format_chat_response(respuesta)
                    
#                     return JsonResponse({
#                         'type': 'html',
#                         'mensaje': formatted_response
#                     })
#                 else:
#                     logger.error(f"Respuesta inesperada: {respuesta}")
#                     return JsonResponse({
#                         'type': 'error',
#                         'mensaje': 'Formato de respuesta no válido'
#                     }, status=400)

#             else:
#                 logger.error(f"Error en la API de FastAPI: {response.status_code}")
#                 return JsonResponse({
#                     'type': 'error',
#                     'mensaje': 'Error al procesar tu solicitud'
#                 }, status=500)

#         except json.JSONDecodeError as e:
#             logger.error(f"Error decodificando JSON: {str(e)}")
#             return JsonResponse({
#                 'type': 'error',
#                 'mensaje': 'Error en el formato de la solicitud'
#             }, status=400)
#         except Exception as e:
#             logger.error(f"Error general: {str(e)}")
#             return JsonResponse({
#                 'type': 'error',
#                 'mensaje': 'Error interno del servidor'
#             }, status=500)

#     return JsonResponse({
#         'type': 'error',
#         'mensaje': 'Método no permitido'
#     }, status=405)

@login_required
def api_chat(request):
    if request.method == 'POST':
        try:
            # Leer el JSON enviado desde el frontend
            body_unicode = request.body.decode('utf-8')
            body = json.loads(body_unicode)
            mensaje = body.get('mensaje')

            # Verificar si el mensaje se recibió correctamente
            logger.info(f"Mensaje recibido: {mensaje}")

            # Realizar la petición a la API de FastAPI
            response = requests.post(
                'http://192.241.155.252:8000/recommendations/chat',
                json={'text': mensaje},
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                # Obtener la respuesta y guardar la conversación
                html_response = response.text
                
                # Guardar la conversación
                conversation = Conversation.objects.create(
                    user=request.user,
                    input=mensaje,
                    output=html_response
                )

                return JsonResponse({
                    'type': 'html',
                    'mensaje': html_response
                })
            else:
                logger.error(f"Error en la API de FastAPI: {response.status_code}")
                return JsonResponse({
                    'type': 'error',
                    'mensaje': 'Error al procesar tu solicitud'
                }, status=500)

        except json.JSONDecodeError as e:
            logger.error(f"Error decodificando JSON: {str(e)}")
            return JsonResponse({
                'type': 'error',
                'mensaje': 'Error en el formato de la solicitud'
            }, status=400)
        except Exception as e:
            logger.error(f"Error general: {str(e)}")
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
    return render(request, 'index.html')

@csrf_protect  # Protección CSRF habilitada
def api_view(request):
    if request.method == "POST":
        user_input = request.POST.get('mensaje', '')
        try:
            response = requests.post('http://192.241.155.252:8000/recommendations/chat', json={'text': user_input})
            if response.status_code == 200:
                response_data = response.json()
                if 'response' in response_data:
                    return JsonResponse({'mensaje': response_data['response']})
                else:
                    logger.error('Respuesta inesperada desde FastAPI: %s', response_data)
                    return JsonResponse({'error': 'Respuesta inesperada desde el servicio de chat'}, status=500)
            else:
                logger.error('Error de estado desde FastAPI: %s', response.status_code)
                return JsonResponse({'error': 'Error con el servicio de chat'}, status=response.status_code)
        except requests.exceptions.RequestException as e:
            logger.error('Excepción al conectar con FastAPI: %s', e)
            return JsonResponse({'error': 'Error de conexión con el servicio de chat'}, status=500)
    else:
        return JsonResponse({'error': 'Método no permitido'}, status=405)


@csrf_protect
def register(request):
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
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, 'Inicio de sesión exitoso.')
            return redirect('index')  # Redirige a la página principal o la que prefieras
        else:
            messages.error(request, 'Usuario o contraseña incorrecta.')
            return redirect('login')
    return render(request, 'login.html')


def user_logout(request):
    logout(request)
    messages.success(request, 'Sesión cerrada exitosamente.')
    return redirect('login')

@csrf_exempt
def save_vectorization(request):
    if request.method == 'POST':
        try:
            logger.info(f"Request body: {request.body}")
            data = json.loads(request.body)
            logger.info(f"Data received: {data}")
            for item in data:
                Chunk.objects.create(
                    document_id=item['document_id'],
                    content=item['content'],
                    embedding=item['embedding']
                )
            return JsonResponse({"status": "success"})
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return JsonResponse({"status": "fail", "error": str(e)}, status=400)
    return JsonResponse({"status": "fail"}, status=400)

def debug_media(request):
    """Vista de debug para verificar la configuración de media"""
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
    return JsonResponse({"status": "healthy"})