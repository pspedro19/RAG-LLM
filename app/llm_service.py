"""
Servicio LLM Unificado - Soporte para múltiples proveedores con fallback automático

Este módulo proporciona una interfaz unificada para trabajar con diferentes
proveedores de LLM (OpenAI, Claude/Anthropic) con fallback automático.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Proveedores de LLM soportados"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NONE = "none"


class UnifiedLLMService:
    """
    Servicio unificado de LLM con soporte para múltiples proveedores y fallback automático.

    Características:
    - Soporte para OpenAI y Claude (Anthropic)
    - Fallback automático cuando el proveedor principal no está disponible
    - Interfaz unificada para chat completions
    - Manejo de errores y reintentos
    """

    def __init__(self):
        """Inicializa el servicio de LLM con detección automática de proveedores"""
        self.openai_client = None
        self.anthropic_client = None
        self.primary_provider = LLMProvider.NONE
        self.fallback_provider = LLMProvider.NONE

        # Cargar clientes disponibles
        self._initialize_clients()

        # Determinar proveedor primario y fallback
        self._determine_providers()

        logger.info(f"LLM Service inicializado - Primario: {self.primary_provider}, Fallback: {self.fallback_provider}")

    def _initialize_clients(self):
        """Inicializa los clientes de LLM disponibles"""
        # Intentar inicializar OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(api_key=openai_key)
                logger.info("✓ Cliente OpenAI inicializado")
            except Exception as e:
                logger.warning(f"No se pudo inicializar OpenAI: {e}")

        # Intentar inicializar Anthropic (Claude)
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                from anthropic import AsyncAnthropic
                self.anthropic_client = AsyncAnthropic(api_key=anthropic_key)
                logger.info("✓ Cliente Anthropic (Claude) inicializado")
            except Exception as e:
                logger.warning(f"No se pudo inicializar Anthropic: {e}")

    def _determine_providers(self):
        """Determina el proveedor primario y fallback basado en disponibilidad"""
        # Prioridad: OpenAI primero, luego Claude
        if self.openai_client:
            self.primary_provider = LLMProvider.OPENAI
            if self.anthropic_client:
                self.fallback_provider = LLMProvider.ANTHROPIC
        elif self.anthropic_client:
            self.primary_provider = LLMProvider.ANTHROPIC
        else:
            logger.error("⚠️ ADVERTENCIA: No hay proveedores de LLM disponibles")
            self.primary_provider = LLMProvider.NONE

    def is_available(self) -> bool:
        """Verifica si hay al menos un proveedor disponible"""
        return self.primary_provider != LLMProvider.NONE

    def _map_model_name(self, model: str, provider: LLMProvider) -> str:
        """Mapea nombres de modelos entre proveedores"""
        # Mapeo de modelos OpenAI a Claude equivalentes
        openai_to_claude = {
            "gpt-4-turbo": "claude-3-5-sonnet-20241022",
            "gpt-4": "claude-3-5-sonnet-20241022",
            "gpt-3.5-turbo": "claude-3-haiku-20240307",
        }

        # Mapeo inverso
        claude_to_openai = {v: k for k, v in openai_to_claude.items()}

        if provider == LLMProvider.ANTHROPIC:
            # Si el modelo es de OpenAI, mapearlo a Claude
            return openai_to_claude.get(model, "claude-3-5-sonnet-20241022")
        elif provider == LLMProvider.OPENAI:
            # Si el modelo es de Claude, mapearlo a OpenAI
            return claude_to_openai.get(model, "gpt-4-turbo")

        return model

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Realiza una chat completion con fallback automático entre proveedores.

        Args:
            messages: Lista de mensajes en formato OpenAI
            model: Nombre del modelo a usar
            temperature: Temperatura para sampling
            max_tokens: Máximo de tokens a generar
            response_format: Formato de respuesta (ej: {"type": "json_object"})
            **kwargs: Argumentos adicionales específicos del proveedor

        Returns:
            Dict con la respuesta en formato unificado
        """
        if not self.is_available():
            raise Exception("No hay proveedores de LLM disponibles")

        # Intentar con proveedor primario
        try:
            return await self._call_provider(
                provider=self.primary_provider,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Error con proveedor primario ({self.primary_provider}): {e}")

            # Intentar fallback si está disponible
            if self.fallback_provider != LLMProvider.NONE:
                logger.info(f"Intentando con proveedor fallback ({self.fallback_provider})...")
                try:
                    return await self._call_provider(
                        provider=self.fallback_provider,
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                        **kwargs
                    )
                except Exception as fallback_error:
                    logger.error(f"Error con fallback ({self.fallback_provider}): {fallback_error}")
                    raise Exception(f"Todos los proveedores fallaron. Primario: {e}, Fallback: {fallback_error}")
            else:
                # No hay fallback disponible
                raise

    async def _call_provider(
        self,
        provider: LLMProvider,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Llama al proveedor específico y retorna respuesta en formato unificado"""

        if provider == LLMProvider.OPENAI:
            return await self._call_openai(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **kwargs
            )
        elif provider == LLMProvider.ANTHROPIC:
            return await self._call_anthropic(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            raise Exception(f"Proveedor {provider} no soportado")

    async def _call_openai(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Llama a OpenAI API"""
        logger.debug(f"Llamando a OpenAI con modelo {model}")

        call_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            call_params["max_tokens"] = max_tokens

        if response_format:
            call_params["response_format"] = response_format

        # Agregar kwargs adicionales
        call_params.update(kwargs)

        response = await self.openai_client.chat.completions.create(**call_params)

        # Formato unificado
        return {
            "content": response.choices[0].message.content,
            "provider": LLMProvider.OPENAI,
            "model": model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "raw_response": response
        }

    async def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Llama a Anthropic (Claude) API"""
        # Mapear modelo a equivalente de Claude
        claude_model = self._map_model_name(model, LLMProvider.ANTHROPIC)
        logger.debug(f"Llamando a Claude con modelo {claude_model}")

        # Convertir formato de mensajes de OpenAI a Claude
        # Claude requiere separar system message del resto
        system_message = None
        claude_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        call_params = {
            "model": claude_model,
            "messages": claude_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 2048,  # Claude requiere max_tokens
        }

        if system_message:
            call_params["system"] = system_message

        # Agregar kwargs adicionales
        call_params.update(kwargs)

        response = await self.anthropic_client.messages.create(**call_params)

        # Extraer contenido del primer bloque de contenido
        content = ""
        if response.content:
            content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])

        # Formato unificado
        return {
            "content": content,
            "provider": LLMProvider.ANTHROPIC,
            "model": claude_model,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            "raw_response": response
        }

    async def chat_completion_with_vision(
        self,
        text: str,
        image_data: str,  # Base64 encoded image
        image_type: str = "image/jpeg",
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Realiza una chat completion con soporte de visión (imagen + texto).

        Args:
            text: Texto de la consulta
            image_data: Imagen codificada en base64
            image_type: Tipo MIME de la imagen (ej: image/jpeg, image/png)
            model: Nombre del modelo a usar
            temperature: Temperatura para sampling
            max_tokens: Máximo de tokens a generar
            **kwargs: Argumentos adicionales

        Returns:
            Dict con la respuesta en formato unificado
        """
        if not self.is_available():
            raise Exception("No hay proveedores de LLM disponibles")

        # Intentar con proveedor primario
        try:
            return await self._call_provider_vision(
                provider=self.primary_provider,
                text=text,
                image_data=image_data,
                image_type=image_type,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            logger.warning(f"Error con proveedor primario ({self.primary_provider}): {e}")

            # Intentar fallback si está disponible
            if self.fallback_provider != LLMProvider.NONE:
                logger.info(f"Intentando visión con proveedor fallback ({self.fallback_provider})...")
                try:
                    return await self._call_provider_vision(
                        provider=self.fallback_provider,
                        text=text,
                        image_data=image_data,
                        image_type=image_type,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                except Exception as fallback_error:
                    logger.error(f"Error con fallback visión ({self.fallback_provider}): {fallback_error}")
                    raise Exception(f"Todos los proveedores fallaron. Primario: {e}, Fallback: {fallback_error}")
            else:
                raise

    async def _call_provider_vision(
        self,
        provider: LLMProvider,
        text: str,
        image_data: str,
        image_type: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Llama al proveedor específico con soporte de visión"""

        if provider == LLMProvider.OPENAI:
            return await self._call_openai_vision(
                text=text,
                image_data=image_data,
                image_type=image_type,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        elif provider == LLMProvider.ANTHROPIC:
            return await self._call_anthropic_vision(
                text=text,
                image_data=image_data,
                image_type=image_type,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            raise Exception(f"Proveedor {provider} no soporta visión")

    async def _call_openai_vision(
        self,
        text: str,
        image_data: str,
        image_type: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Llama a OpenAI GPT-4 Vision"""
        # Usar GPT-4 Vision si se especifica turbo
        vision_model = "gpt-4o" if "turbo" in model or "4" in model else model
        logger.debug(f"Llamando a OpenAI Vision con modelo {vision_model}")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_type};base64,{image_data}"
                        }
                    }
                ]
            }
        ]

        call_params = {
            "model": vision_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 1000,
        }
        call_params.update(kwargs)

        response = await self.openai_client.chat.completions.create(**call_params)

        return {
            "content": response.choices[0].message.content,
            "provider": LLMProvider.OPENAI,
            "model": vision_model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "raw_response": response
        }

    async def _call_anthropic_vision(
        self,
        text: str,
        image_data: str,
        image_type: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Llama a Claude 3 Vision"""
        # Claude 3.5 Sonnet tiene capacidades de visión
        claude_model = "claude-3-5-sonnet-20241022"
        logger.debug(f"Llamando a Claude Vision con modelo {claude_model}")

        # Determinar media_type correcto para Claude
        media_type = image_type  # "image/jpeg", "image/png", etc.

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": text
                    }
                ],
            }
        ]

        call_params = {
            "model": claude_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 1024,
        }
        call_params.update(kwargs)

        response = await self.anthropic_client.messages.create(**call_params)

        # Extraer contenido
        content = ""
        if response.content:
            content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])

        return {
            "content": content,
            "provider": LLMProvider.ANTHROPIC,
            "model": claude_model,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            "raw_response": response
        }

    async def embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-3-small") -> List[float]:
        """
        Genera embeddings (solo soportado con OpenAI por ahora)

        Args:
            text: Texto o lista de textos
            model: Modelo de embeddings

        Returns:
            Lista de embeddings
        """
        if not self.openai_client:
            raise Exception("Embeddings solo están disponibles con OpenAI")

        response = await self.openai_client.embeddings.create(
            model=model,
            input=text
        )

        if isinstance(text, list):
            return [item.embedding for item in response.data]
        else:
            return response.data[0].embedding

    def get_available_providers(self) -> List[str]:
        """Retorna lista de proveedores disponibles"""
        providers = []
        if self.openai_client:
            providers.append(LLMProvider.OPENAI)
        if self.anthropic_client:
            providers.append(LLMProvider.ANTHROPIC)
        return providers

    def get_status(self) -> Dict[str, Any]:
        """Retorna el estado del servicio"""
        return {
            "available_providers": self.get_available_providers(),
            "primary_provider": self.primary_provider,
            "fallback_provider": self.fallback_provider,
            "is_available": self.is_available()
        }


# Singleton global
_llm_service = None

def get_llm_service() -> UnifiedLLMService:
    """Obtiene la instancia singleton del servicio LLM"""
    global _llm_service
    if _llm_service is None:
        _llm_service = UnifiedLLMService()
    return _llm_service
