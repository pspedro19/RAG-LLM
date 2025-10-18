"""
Security Module - Rate Limiting, API Key Auth, Input Validation
"""

import os
import time
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import re
import logging

logger = logging.getLogger(__name__)

# ==================================
# Rate Limiter Configuration
# ==================================
limiter = Limiter(key_func=get_remote_address)


# ==================================
# API Key Authentication (Optional)
# ==================================
security = HTTPBearer(auto_error=False)

VALID_API_KEYS = set(
    filter(None, os.environ.get("API_KEYS", "").split(","))
)

def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> bool:
    """
    Verifica API key si está configurada.
    Si no hay API keys configuradas, permite acceso.
    """
    # Si no hay keys configuradas, sistema abierto
    if not VALID_API_KEYS:
        return True

    # Si hay keys configuradas pero no se proporcionó credencial
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verificar que la key sea válida
    if credentials.credentials not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempt: {credentials.credentials[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


# ==================================
# Input Sanitization
# ==================================
MAX_QUERY_LENGTH = 2000
ALLOWED_CHARS_PATTERN = re.compile(r'^[\w\s\-.,;:!?¿¡áéíóúñÁÉÍÓÚÑ()\'\"@#$%&+=\[\]{}/<>]+$', re.UNICODE)

def sanitize_input(text: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """
    Sanitiza y valida input del usuario.

    Args:
        text: Texto a sanitizar
        max_length: Longitud máxima permitida

    Returns:
        Texto sanitizado

    Raises:
        HTTPException: Si el input es inválido
    """
    # Validar que no esté vacío
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )

    # Truncar si es muy largo
    text = text.strip()[:max_length]

    # Validar caracteres permitidos (prevenir inyecciones)
    if not ALLOWED_CHARS_PATTERN.match(text):
        # Remover caracteres no permitidos en lugar de rechazar
        text = re.sub(r'[^\w\s\-.,;:!?¿¡áéíóúñÁÉÍÓÚÑ()\'\"@#$%&+=\[\]{}/<>]', '', text, flags=re.UNICODE)
        logger.warning(f"Input sanitized: removed special characters")

    return text


# ==================================
# Output Validation
# ==================================
def validate_output(response: str, max_length: int = 10000) -> str:
    """
    Valida que el output sea seguro antes de enviarlo.

    Args:
        response: Respuesta a validar
        max_length: Longitud máxima

    Returns:
        Respuesta validada
    """
    # Truncar si es muy larga
    if len(response) > max_length:
        logger.warning(f"Response truncated: {len(response)} > {max_length}")
        response = response[:max_length] + "... [truncated]"

    # Remover potenciales scripts o HTML malicioso
    response = response.replace("<script>", "&lt;script&gt;")
    response = response.replace("</script>", "&lt;/script&gt;")

    return response


# ==================================
# Security Headers Middleware
# ==================================
def add_security_headers(response):
    """Añade headers de seguridad a las respuestas"""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ==================================
# Rate Limiting Decorators
# ==================================
# Para usar en endpoints:
# @limiter.limit("5/minute")  # 5 requests por minuto
# @limiter.limit("100/hour")  # 100 requests por hora
# @limiter.limit("1000/day")  # 1000 requests por día
