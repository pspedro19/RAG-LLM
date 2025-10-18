"""
Webhook Service - Notificación a sistemas externos

Requisito del proyecto: "Acción final disparada automáticamente (webhook)"
"""

import logging
import os
import asyncio
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class WebhookService:
    """Servicio para disparar webhooks a sistemas externos"""

    def __init__(self, default_url: Optional[str] = None, timeout: int = 10):
        """
        Inicializa el servicio de webhooks

        Args:
            default_url: URL por defecto para webhooks
            timeout: Timeout en segundos para requests HTTP
        """
        self.default_url = default_url or os.environ.get("WEBHOOK_URL")
        self.timeout = timeout
        self.stats = {
            "total_webhooks": 0,
            "successful": 0,
            "failed": 0
        }

        if self.default_url:
            logger.info(f"✅ WebhookService inicializado con URL: {self.default_url}")
        else:
            logger.warning("⚠️ WEBHOOK_URL no configurado. Webhooks se registrarán en logs.")

    async def trigger(
        self,
        data: Dict[str, Any],
        url: Optional[str] = None,
        event_type: str = "generic"
    ) -> bool:
        """
        Dispara un webhook con los datos proporcionados

        Args:
            data: Datos a enviar en el webhook
            url: URL del webhook (usa default_url si no se especifica)
            event_type: Tipo de evento para categorización

        Returns:
            True si el webhook se disparó exitosamente
        """
        target_url = url or self.default_url
        self.stats["total_webhooks"] += 1

        if not target_url:
            # Modo fallback: registrar en logs
            logger.info(f"🔔 [WEBHOOK FALLBACK] Evento: {event_type}")
            logger.info(f"   Datos: {data}")
            self.stats["successful"] += 1
            return True

        try:
            import aiohttp

            # Agregar metadata al payload
            payload = {
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    target_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        logger.info(f"✅ Webhook disparado exitosamente: {event_type} -> {target_url}")
                        self.stats["successful"] += 1
                        return True
                    else:
                        logger.warning(f"⚠️ Webhook respondió con status {response.status}: {target_url}")
                        self.stats["failed"] += 1
                        return False

        except asyncio.TimeoutError:
            logger.error(f"⏱️ Timeout disparando webhook a {target_url} (>{self.timeout}s)")
            self.stats["failed"] += 1
            return False
        except ImportError:
            logger.error("📦 aiohttp no instalado. Instale con: pip install aiohttp")
            self.stats["failed"] += 1
            return False
        except Exception as e:
            logger.error(f"❌ Error disparando webhook a {target_url}: {e}")
            self.stats["failed"] += 1
            return False

    async def trigger_itinerary_generated(
        self,
        conversation_id: str,
        query: str,
        itinerary: str,
        user_email: Optional[str] = None
    ) -> bool:
        """
        Webhook especializado: Itinerario generado

        Args:
            conversation_id: ID de la conversación
            query: Consulta original
            itinerary: Itinerario generado
            user_email: Email del usuario (opcional)

        Returns:
            True si el webhook se disparó exitosamente
        """
        data = {
            "conversation_id": conversation_id,
            "query": query,
            "itinerary_preview": itinerary[:200] + "..." if len(itinerary) > 200 else itinerary,
            "itinerary_length": len(itinerary),
            "user_email": user_email
        }

        return await self.trigger(data, event_type="itinerary_generated")

    async def trigger_query_completed(
        self,
        conversation_id: str,
        query: str,
        query_type: str,
        response_preview: str,
        processing_time: float,
        tokens_used: Optional[int] = None
    ) -> bool:
        """
        Webhook especializado: Consulta completada

        Args:
            conversation_id: ID de la conversación
            query: Consulta original
            query_type: Tipo de consulta (conversacional/informacion/itinerario)
            response_preview: Preview de la respuesta
            processing_time: Tiempo de procesamiento en segundos
            tokens_used: Tokens consumidos (opcional)

        Returns:
            True si el webhook se disparó exitosamente
        """
        data = {
            "conversation_id": conversation_id,
            "query": query,
            "query_type": query_type,
            "response_preview": response_preview,
            "processing_time": processing_time,
            "tokens_used": tokens_used
        }

        return await self.trigger(data, event_type="query_completed")

    async def trigger_error(
        self,
        conversation_id: str,
        error_type: str,
        error_message: str,
        query: Optional[str] = None
    ) -> bool:
        """
        Webhook especializado: Error ocurrido

        Args:
            conversation_id: ID de la conversación
            error_type: Tipo de error
            error_message: Mensaje de error
            query: Consulta que causó el error (opcional)

        Returns:
            True si el webhook se disparó exitosamente
        """
        data = {
            "conversation_id": conversation_id,
            "error_type": error_type,
            "error_message": error_message,
            "query": query
        }

        return await self.trigger(data, event_type="error")

    def get_stats(self) -> Dict[str, int]:
        """Retorna estadísticas de webhooks"""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful"] / self.stats["total_webhooks"] * 100
                if self.stats["total_webhooks"] > 0 else 0
            )
        }
