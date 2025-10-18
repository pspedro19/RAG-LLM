"""
Notifications Module - Acciones automatizadas

Proporciona capacidades de notificación mediante:
- Email (SendGrid)
- Webhooks
- Logging externo

Requisito del proyecto: "Acción final disparada automáticamente"
"""

from .email_service import EmailService
from .webhook_service import WebhookService

__all__ = ['EmailService', 'WebhookService']
