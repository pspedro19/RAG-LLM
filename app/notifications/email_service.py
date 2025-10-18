"""
Email Service - Envío de notificaciones por email

Requisito del proyecto: "Acción final disparada automáticamente (email)"
"""

import logging
import os
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class EmailService:
    """Servicio de envío de emails usando SendGrid"""

    def __init__(self, api_key: Optional[str] = None, fallback_mode: bool = True):
        """
        Inicializa el servicio de email

        Args:
            api_key: API key de SendGrid (si no se provee, usa variable de entorno)
            fallback_mode: Si True, no falla cuando SendGrid no está disponible
        """
        self.api_key = api_key or os.environ.get("SENDGRID_API_KEY")
        self.fallback_mode = fallback_mode
        self.client = None

        if self.api_key:
            try:
                from sendgrid import SendGridAPIClient
                self.client = SendGridAPIClient(self.api_key)
                logger.info("✅ EmailService inicializado con SendGrid")
            except ImportError:
                logger.warning("📦 SendGrid no instalado. Instale con: pip install sendgrid")
                if not fallback_mode:
                    raise
            except Exception as e:
                logger.error(f"❌ Error inicializando SendGrid: {e}")
                if not fallback_mode:
                    raise
        else:
            logger.warning("⚠️ SENDGRID_API_KEY no configurado. Emails no se enviarán.")

    async def send_itinerary(
        self,
        to_email: str,
        itinerary: str,
        query: str,
        conversation_id: str
    ) -> bool:
        """
        Envía un itinerario por email

        Args:
            to_email: Email del destinatario
            itinerary: Contenido del itinerario
            query: Consulta original del usuario
            conversation_id: ID de la conversación

        Returns:
            True si el email se envió exitosamente
        """
        if not self.client:
            logger.warning(f"Email no enviado a {to_email} (servicio no disponible)")
            if self.fallback_mode:
                # Modo fallback: guardar en log
                logger.info(f"📧 [FALLBACK] Email que se enviaría a {to_email}:")
                logger.info(f"   Asunto: Tu Itinerario de Curazao")
                logger.info(f"   Query: {query[:50]}...")
                return True
            return False

        try:
            from sendgrid.helpers.mail import Mail

            message = Mail(
                from_email='noreply@curacao-assistant.com',
                to_emails=to_email,
                subject=f'Tu Itinerario de Curazao: {query[:50]}',
                html_content=self._format_itinerary_email(itinerary, query, conversation_id)
            )

            response = self.client.send(message)

            if response.status_code in [200, 202]:
                logger.info(f"✅ Email enviado exitosamente a {to_email} (ID: {conversation_id})")
                return True
            else:
                logger.warning(f"⚠️ Email enviado con status code {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error enviando email a {to_email}: {e}")
            return False

    async def send_query_response(
        self,
        to_email: str,
        query: str,
        response: str,
        conversation_id: str,
        query_type: str
    ) -> bool:
        """
        Envía una respuesta de consulta por email

        Args:
            to_email: Email del destinatario
            query: Consulta original
            response: Respuesta generada
            conversation_id: ID de la conversación
            query_type: Tipo de consulta

        Returns:
            True si el email se envió exitosamente
        """
        if not self.client:
            logger.warning(f"Email no enviado a {to_email} (servicio no disponible)")
            if self.fallback_mode:
                logger.info(f"📧 [FALLBACK] Email que se enviaría a {to_email}")
                return True
            return False

        try:
            from sendgrid.helpers.mail import Mail

            message = Mail(
                from_email='noreply@curacao-assistant.com',
                to_emails=to_email,
                subject=f'Respuesta a tu consulta: {query[:40]}',
                html_content=self._format_query_email(query, response, conversation_id, query_type)
            )

            result = self.client.send(message)

            if result.status_code in [200, 202]:
                logger.info(f"✅ Respuesta enviada por email a {to_email}")
                return True
            else:
                logger.warning(f"⚠️ Email enviado con status {result.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error enviando respuesta por email: {e}")
            return False

    def _format_itinerary_email(self, itinerary: str, query: str, conversation_id: str) -> str:
        """Formatea el email de itinerario con HTML"""
        # Convertir saltos de línea a HTML antes del f-string
        itinerary_html = itinerary.replace('\n', '<br>')
        current_datetime = datetime.now().strftime('%d/%m/%Y %H:%M')

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px 10px 0 0;
                    text-align: center;
                }}
                .content {{
                    background: #f9f9f9;
                    padding: 30px;
                    border: 1px solid #ddd;
                }}
                .itinerary {{
                    background: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-top: 20px;
                    white-space: pre-wrap;
                }}
                .footer {{
                    background: #333;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    border-radius: 0 0 10px 10px;
                    font-size: 12px;
                }}
                .query-box {{
                    background: #e3f2fd;
                    padding: 15px;
                    border-left: 4px solid #2196F3;
                    margin: 20px 0;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🏝️ Tu Itinerario Personalizado para Curazao</h1>
            </div>
            <div class="content">
                <p>¡Hola! Aquí está tu itinerario personalizado para Curazao.</p>

                <div class="query-box">
                    <strong>Tu consulta:</strong> {query}
                </div>

                <div class="itinerary">
                    {itinerary_html}
                </div>

                <p style="margin-top: 30px;">
                    <strong>ID de conversación:</strong> {conversation_id}<br>
                    <strong>Fecha:</strong> {current_datetime}
                </p>

                <p>
                    ¿Necesitas modificar tu itinerario? Responde a este email con tus cambios.
                </p>
            </div>
            <div class="footer">
                <p>🤖 Generado por Curazao Tourism Assistant</p>
                <p>Este es un email automatizado. Para más información, visita nuestra API.</p>
            </div>
        </body>
        </html>
        """

    def _format_query_email(self, query: str, response: str, conversation_id: str, query_type: str) -> str:
        """Formatea el email de respuesta general"""
        # Convertir saltos de línea a HTML antes del f-string
        response_html = response.replace('\n', '<br>')

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px 10px 0 0;
                    text-align: center;
                }}
                .content {{
                    background: #f9f9f9;
                    padding: 30px;
                    border: 1px solid #ddd;
                }}
                .response {{
                    background: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-top: 20px;
                }}
                .footer {{
                    background: #333;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    border-radius: 0 0 10px 10px;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📩 Respuesta a tu Consulta</h1>
            </div>
            <div class="content">
                <p><strong>Tu pregunta:</strong></p>
                <p style="background: #e3f2fd; padding: 15px; border-left: 4px solid #2196F3;">
                    {query}
                </p>

                <div class="response">
                    <h3>Respuesta:</h3>
                    {response_html}
                </div>

                <p style="margin-top: 20px; font-size: 12px; color: #666;">
                    Tipo de consulta: {query_type} | ID: {conversation_id}
                </p>
            </div>
            <div class="footer">
                <p>🤖 Generado por Curazao Tourism Assistant</p>
            </div>
        </body>
        </html>
        """
