#!/usr/bin/env python3
"""
Enhanced RAG Chat - Interfaz conversacional natural con RAG + LLM

Este script proporciona una experiencia de chat natural, integrando el sistema RAG
con modelos de lenguaje externos (OpenAI, Claude) para generar respuestas fluidas
sin mostrar información técnica al usuario.

Características:
1. Conversación natural, sin exponer logs al usuario
2. Uso inteligente del RAG solo cuando es necesario
3. Respuestas resumidas y bien formateadas
4. Detección de preguntas que requieren consulta a la base de conocimiento

Uso:
    python enhanced_rag_chat.py [--model openai|claude] [--top-k 5]
"""

import asyncio
import argparse
import logging
import os
import sys
import textwrap
import shutil
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  # Cargar variables de .env

# Añadir directorio padre al path para importar módulos
sys.path.insert(0, os.path.abspath('..'))

# Configurar logging silencioso (los logs van a un archivo, no a la consola)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_rag_chat.log"),
    ]
)

logger = logging.getLogger("enhanced-rag-chat")

# Suprimir logs de baja prioridad
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("app2").setLevel(logging.WARNING)

# Importar componentes necesarios
from app2.core.config.config import Config
from app2.core.pipelines.search_pipeline import SearchPipeline

# Importar APIs de LLM (si están disponibles)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# Colores ANSI para la terminal
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

# Verificar si la terminal soporta colores
def supports_color():
    """Verifica si la terminal soporta colores ANSI."""
    if os.name == 'nt':  # Windows
        return False  # Por defecto, desactivar en Windows
    
    # Verificar variables de entorno
    if 'NO_COLOR' in os.environ:
        return False
    
    if not sys.stdout.isatty():
        return False
    
    return True

# Deshabilitar colores si no hay soporte
if not supports_color():
    for attr in dir(Colors):
        if not attr.startswith('__'):
            setattr(Colors, attr, "")

class EnhancedRAGChat:
    """Interfaz conversacional mejorada que integra RAG con LLMs externos."""
    
    def __init__(
        self, 
        model: str = 'openai',
        top_k: int = 5,
        use_localhost: bool = True,
        max_conversation_history: int = 10
    ):
        """
        Inicializa el chat mejorado.
        
        Args:
            model: Modelo a utilizar ('openai' o 'claude')
            top_k: Número de resultados a devolver del RAG
            use_localhost: Si es True, usa localhost en lugar de postgres_pgvector
            max_conversation_history: Máximo número de mensajes a mantener en historial
        """
        self.config = Config()
        
        # Usar localhost para la base de datos si se especifica
        if use_localhost:
            logger.info("Usando localhost para conectar a PostgreSQL")
            self.config.DB_HOST = "localhost"
        
        # Configuración de búsqueda
        self.top_k = top_k
        self.model_name = model
        self.max_conversation_history = max_conversation_history
        
        # Inicializar componentes
        self.search_pipeline = None
        
        # Historial de conversación
        self.conversation = []
        
        # Obtener tamaño de terminal
        self.term_width, self.term_height = shutil.get_terminal_size()
        
        # Validar disponibilidad de APIs
        self.openai_available = self._check_openai_availability()
        self.claude_available = self._check_claude_availability()
        
        # Determinar el modelo a usar según disponibilidad
        self.active_model = self._determine_active_model()
        
        logger.info(f"Enhanced RAG Chat inicializado (modelo: {self.active_model}, top-k: {top_k})")
    
    def _check_openai_availability(self) -> bool:
        """Verifica si OpenAI está disponible y configurado."""
        if not OPENAI_AVAILABLE:
            logger.warning("Biblioteca OpenAI no instalada")
            return False
        
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if not api_key:
            logger.warning("OPENAI_API_KEY no encontrada en variables de entorno")
            return False
        
        # Inicializar cliente
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            logger.info("Cliente OpenAI inicializado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error inicializando cliente OpenAI: {e}")
            return False
    
    def _check_claude_availability(self) -> bool:
        """Verifica si Claude está disponible y configurado."""
        if not CLAUDE_AVAILABLE:
            logger.warning("Biblioteca Anthropic no instalada")
            return False
        
        api_key = os.environ.get("ANTHROPIC_API_KEY", None)
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY no encontrada en variables de entorno")
            return False
        
        # Inicializar cliente
        try:
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            logger.info("Cliente Claude inicializado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error inicializando cliente Claude: {e}")
            return False
    
    def _determine_active_model(self) -> str:
        """Determina qué modelo usar basado en disponibilidad."""
        # Verificar primera preferencia
        if self.model_name == 'openai' and self.openai_available:
            return 'openai'
        elif self.model_name == 'claude' and self.claude_available:
            return 'claude'
        
        # Si la primera preferencia no está disponible, intentar alternativa
        if self.openai_available:
            return 'openai'
        elif self.claude_available:
            return 'claude'
        
        # Si ninguno está disponible, usar solo RAG
        logger.warning("Ningún LLM externo disponible. Usando solo RAG.")
        return 'rag-only'
    
    async def initialize(self) -> bool:
        """Inicializa los componentes del sistema RAG."""
        logger.info("Inicializando componentes...")
        
        try:
            # Crear pipeline de búsqueda
            self.search_pipeline = SearchPipeline(config=self.config)
            logger.info("Componentes inicializados correctamente")
            return True
        except Exception as e:
            logger.error(f"Error inicializando componentes: {e}")
            return False
    
    async def close(self):
        """Cierra conexiones y limpia recursos."""
        if self.search_pipeline:
            await self.search_pipeline.close()
        logger.info("Recursos de Enhanced RAG Chat cerrados")
    
    async def _perform_rag_search(self, query: str) -> Dict[str, Any]:
        """
        Realiza una búsqueda en el sistema RAG.
        
        Args:
            query: Consulta a realizar
            
        Returns:
            Resultados de la búsqueda o diccionario vacío si falla
        """
        if not self.search_pipeline:
            logger.error("Pipeline de búsqueda no inicializado")
            return {}
        
        logger.info(f"Buscando en RAG: '{query}'")
        
        try:
            # Realizar búsqueda
            results = await self.search_pipeline.search(
                query_text=query,
                mode='hybrid',
                top_k=self.top_k,
                strategy='relevance'
            )
            
            logger.info(f"Búsqueda completada: {len(results.get('chunks', []))} chunks encontrados")
            return results
        except Exception as e:
            logger.error(f"Error en búsqueda RAG: {e}")
            return {}
    
    def _create_prompt_with_rag_context(self, query: str, rag_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea un prompt para el LLM incluyendo contexto del RAG.
        
        Args:
            query: Consulta del usuario
            rag_results: Resultados del RAG
            
        Returns:
            Prompt formateado para el LLM
        """
        # Obtener contexto
        context = rag_results.get('context', "")
        
        # Crear sistema de mensajes para el historial
        if self.active_model == 'openai':
            # Formato para OpenAI
            messages = []
            
            # Mensaje del sistema
            system_msg = """
            Eres un asistente turístico especializado en Curaçao. Utiliza la información proporcionada 
            para responder preguntas del usuario. Si la información no está en el contexto, 
            indica que no tienes suficiente información. Responde de manera natural y conversacional, 
            sin mencionar términos técnicos como "chunks", "contexto", "faiss", etc. 
            Proporciona respuestas concisas y bien organizadas.
            """
            messages.append({"role": "system", "content": system_msg.strip()})
            
            # Añadir historial de conversación
            for msg in self.conversation:
                messages.append(msg)
            
            # Añadir contexto del RAG
            if context:
                context_msg = f"""
                Aquí hay información relevante sobre la consulta:
                
                {context}
                
                Usa esta información para responder la consulta actual si es relevante.
                """
                messages.append({"role": "system", "content": context_msg.strip()})
            
            # Añadir consulta actual
            messages.append({"role": "user", "content": query})
            
            return messages
        
        elif self.active_model == 'claude':
            # Formato para Claude (un solo mensaje con todo el contexto)
            system_prompt = """
            \n\nHuman: Eres un asistente turístico especializado en Curaçao. Utiliza la información proporcionada 
            para responder preguntas del usuario. Si la información no está en el contexto, 
            indica que no tienes suficiente información. Responde de manera natural y conversacional, 
            sin mencionar términos técnicos como "chunks", "contexto", "faiss", etc. 
            Proporciona respuestas concisas y bien organizadas.
            """
            
            # Añadir historial de conversación
            conversation_text = ""
            for msg in self.conversation:
                if msg["role"] == "user":
                    conversation_text += f"\n\nHuman: {msg['content']}"
                else:
                    conversation_text += f"\n\nAssistant: {msg['content']}"
            
            # Añadir contexto RAG
            context_text = ""
            if context:
                context_text = f"""
                \n\nHuman: Aquí hay información relevante sobre la consulta:
                
                {context}
                
                Usa esta información para responder la consulta actual si es relevante.
                """
            
            # Añadir consulta actual
            query_text = f"\n\nHuman: {query}"
            
            # Combinar todo
            prompt = system_prompt + conversation_text + context_text + query_text + "\n\nAssistant:"
            
            return prompt
        
        else:
            # Formato simple para fallback
            return {"query": query, "context": context}
    
    def _create_simple_prompt(self, query: str) -> Dict[str, Any]:
        """
        Crea un prompt para el LLM sin contexto RAG.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Prompt formateado para el LLM
        """
        if self.active_model == 'openai':
            # Formato para OpenAI
            messages = []
            
            # Mensaje del sistema
            system_msg = """
            Eres un asistente turístico especializado en Curaçao. Responde de manera concisa y natural.
            Si no tienes suficiente información para responder, indícalo claramente y sugiere qué 
            información necesitarías para dar una respuesta más completa.
            """
            messages.append({"role": "system", "content": system_msg.strip()})
            
            # Añadir historial de conversación
            for msg in self.conversation:
                messages.append(msg)
            
            # Añadir consulta actual
            messages.append({"role": "user", "content": query})
            
            return messages
        
        elif self.active_model == 'claude':
            # Formato para Claude
            system_prompt = """
            \n\nHuman: Eres un asistente turístico especializado en Curaçao. Responde de manera concisa y natural.
            Si no tienes suficiente información para responder, indícalo claramente y sugiere qué 
            información necesitarías para dar una respuesta más completa.
            """
            
            # Añadir historial de conversación
            conversation_text = ""
            for msg in self.conversation:
                if msg["role"] == "user":
                    conversation_text += f"\n\nHuman: {msg['content']}"
                else:
                    conversation_text += f"\n\nAssistant: {msg['content']}"
            
            # Añadir consulta actual
            query_text = f"\n\nHuman: {query}"
            
            # Combinar todo
            prompt = system_prompt + conversation_text + query_text + "\n\nAssistant:"
            
            return prompt
        
        else:
            # Formato simple para fallback
            return {"query": query}
    
    async def _query_needs_rag(self, query: str) -> bool:
        """
        Determina si una consulta necesita usar el RAG o puede ser respondida directamente.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            True si la consulta necesita RAG, False si no
        """
        # Consultas genéricas que no necesitan RAG
        generic_greetings = [
            "hola", "hello", "hi", "buenos días", "buenas tardes",
            "cómo estás", "how are you", "saludos", "qué tal",
            "adiós", "bye", "hasta luego", "gracias", "thank you",
        ]
        
        # Verificar si es un saludo simple
        if query.lower().strip().strip("?!.,") in generic_greetings:
            logger.info(f"Consulta genérica detectada, no se usará RAG: '{query}'")
            return False
        
        # Palabras clave que indican que es probable que necesite RAG
        rag_keywords = [
            "curaçao", "curacao", "playas", "beaches", "atracciones", "attractions",
            "hotel", "restaurante", "restaurant", "actividades", "activities",
            "tourism", "turismo", "isla", "island", "caribe", "caribbean",
            "willemstad", "playa", "beach", "museo", "museum", "snorkel",
            "buceo", "diving", "historia", "history", "precio", "price"
        ]
        
        # Verificar si contiene palabras clave relacionadas con Curaçao
        for keyword in rag_keywords:
            if keyword.lower() in query.lower():
                logger.info(f"Consulta relacionada con el dominio, se usará RAG: '{query}'")
                return True
        
        # Para otros mensajes, usar modelo para decidir, pero por ahora simple
        if len(query.split()) > 5:  # Si tiene más de 5 palabras, podría ser una pregunta compleja
            return True
        
        # Por defecto, no usar RAG
        logger.info(f"Consulta no categorizada, no se usará RAG: '{query}'")
        return False
    
    async def _generate_response_with_openai(self, messages: List[Dict[str, str]]) -> str:
        """
        Genera una respuesta usando la API de OpenAI.
        
        Args:
            messages: Lista de mensajes en formato OpenAI
            
        Returns:
            Respuesta generada
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error con OpenAI: {e}")
            return f"Lo siento, ocurrió un error al procesar tu consulta. Detalles: {str(e)}"
    
    async def _generate_response_with_claude(self, prompt: str) -> str:
        """
        Genera una respuesta usando la API de Claude.
        
        Args:
            prompt: Prompt para Claude
            
        Returns:
            Respuesta generada
        """
        try:
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error con Claude: {e}")
            return f"Lo siento, ocurrió un error al procesar tu consulta. Detalles: {str(e)}"
    
    async def _generate_response_with_rag_only(self, query_data: Dict[str, Any]) -> str:
        """
        Genera una respuesta simple usando solo los resultados del RAG.
        
        Args:
            query_data: Datos de la consulta
            
        Returns:
            Respuesta generada
        """
        context = query_data.get("context", "")
        
        if not context:
            return "Lo siento, no pude encontrar información relevante para tu consulta."
        
        # Respuesta muy simple que resume el contexto
        return (
            "Aquí está la información que encontré:\n\n" +
            context +
            "\n\nEsta es la información directa de nuestras fuentes de datos."
        )
    
    async def _process_query(self, query: str) -> str:
        """
        Procesa una consulta del usuario y genera una respuesta.
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Respuesta generada
        """
        # Determinar si la consulta necesita RAG
        needs_rag = await self._query_needs_rag(query)
        
        # Si necesita RAG, realizar búsqueda
        rag_results = {}
        if needs_rag:
            print(f"{Colors.DIM}Buscando información relevante...{Colors.RESET}", flush=True)
            rag_results = await self._perform_rag_search(query)
        
        # Generar prompt según el modelo activo
        if needs_rag and rag_results:
            # Prompt con contexto RAG
            prompt = self._create_prompt_with_rag_context(query, rag_results)
        else:
            # Prompt sin contexto RAG
            prompt = self._create_simple_prompt(query)
        
        # Generar respuesta según el modelo activo
        if self.active_model == 'openai':
            print(f"{Colors.DIM}Generando respuesta con OpenAI...{Colors.RESET}", flush=True)
            response = await self._generate_response_with_openai(prompt)
        elif self.active_model == 'claude':
            print(f"{Colors.DIM}Generando respuesta con Claude...{Colors.RESET}", flush=True)
            response = await self._generate_response_with_claude(prompt)
        else:
            # Fallback a RAG simple
            response = await self._generate_response_with_rag_only(rag_results)
        
        # Actualizar historial de conversación
        self.conversation.append({"role": "user", "content": query})
        self.conversation.append({"role": "assistant", "content": response})
        
        # Truncar historial si excede el máximo
        if len(self.conversation) > self.max_conversation_history * 2:
            self.conversation = self.conversation[-self.max_conversation_history*2:]
        
        return response
    
    def print_welcome(self):
        """Imprime mensaje de bienvenida."""
        title = "Enhanced RAG Chat - Asistente Turístico de Curaçao"
        subtitle = f"Modelo activo: {self.active_model.upper()}"
        
        print("\n" + "=" * self.term_width)
        print(f"{Colors.BOLD}{Colors.GREEN}{title.center(self.term_width)}{Colors.RESET}")
        print(f"{Colors.DIM}{subtitle.center(self.term_width)}{Colors.RESET}")
        print("=" * self.term_width)
        
        print(f"\n{Colors.CYAN}¡Bienvenido al asistente turístico de Curaçao!{Colors.RESET}")
        print("Puedo ayudarte con información sobre la isla, atracciones, playas, restaurantes y más.")
        
        if self.active_model == 'rag-only':
            print(f"\n{Colors.YELLOW}Aviso: No se ha detectado ninguna API key de LLM válida.{Colors.RESET}")
            print(f"{Colors.YELLOW}Se utilizará solo el sistema RAG para las respuestas.{Colors.RESET}")
            print(f"{Colors.YELLOW}Para una experiencia completa, configura OPENAI_API_KEY o ANTHROPIC_API_KEY.{Colors.RESET}")
        
        print("\nEscribe tu consulta y presione Enter para comenzar.")
        print("Escribe '/salir' para terminar la conversación.")
        print("=" * self.term_width + "\n")
    
    def format_response(self, response: str) -> str:
        """
        Formatea una respuesta para mostrar en la terminal.
        
        Args:
            response: Respuesta a formatear
            
        Returns:
            Respuesta formateada
        """
        # Aplicar wrap a las líneas
        lines = []
        for paragraph in response.split('\n'):
            if not paragraph.strip():
                lines.append("")
                continue
            
            # Wrap cada párrafo
            wrapped = textwrap.wrap(paragraph, width=self.term_width-4)
            lines.extend(wrapped)
        
        # Unir con indentación
        formatted = "\n".join(["    " + line for line in lines])
        
        return formatted
    
    async def run(self):
        """Ejecuta el chat interactivo."""
        # Inicializar componentes
        if not await self.initialize():
            print(f"\n{Colors.RED}Error inicializando el sistema. Verifique los logs para más detalles.{Colors.RESET}\n")
            return
        
        # Mostrar bienvenida
        self.print_welcome()
        
        # Loop principal
        try:
            while True:
                # Actualizar tamaño de terminal
                self.term_width, self.term_height = shutil.get_terminal_size()
                
                # Solicitar entrada
                query = input(f"\n{Colors.BOLD}{Colors.GREEN}Tú:{Colors.RESET} ")
                
                # Verificar si está vacío
                if not query.strip():
                    continue
                
                # Verificar si es comando de salida
                if query.lower() in ['/salir', '/exit', '/quit']:
                    print(f"\n{Colors.GREEN}¡Gracias por usar el asistente turístico de Curaçao! ¡Hasta pronto!{Colors.RESET}\n")
                    break
                
                # Procesar consulta
                response = await self._process_query(query)
                
                # Mostrar respuesta
                print(f"\n{Colors.BOLD}{Colors.BLUE}Asistente:{Colors.RESET}")
                formatted_response = self.format_response(response)
                print(formatted_response)
                
        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}¡Hasta pronto!{Colors.RESET}\n")
        except Exception as e:
            logger.error(f"Error inesperado: {e}")
            print(f"\n{Colors.RED}Error inesperado: {e}{Colors.RESET}\n")
        finally:
            # Cerrar recursos
            await self.close()

async def main():
    """Función principal para ejecución desde la línea de comandos."""
    parser = argparse.ArgumentParser(description='Enhanced RAG Chat - Asistente turístico de Curaçao')
    
    parser.add_argument('--model', choices=['openai', 'claude'],
                      default='openai', help='Modelo a utilizar')
    parser.add_argument('--top-k', type=int, default=5,
                      help='Número de resultados a devolver del RAG')
    parser.add_argument('--use-container-name', action='store_true',
                      help='Usar postgres_pgvector en lugar de localhost')
    parser.add_argument('--debug', action='store_true',
                      help='Activar logging de depuración')
    
    args = parser.parse_args()
    
    # Configurar nivel de logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("app2").setLevel(logging.DEBUG)
    
    # Inicializar chat
    chat = EnhancedRAGChat(
        model=args.model,
        top_k=args.top_k,
        use_localhost=not args.use_container_name
    )
    
    # Ejecutar chat
    await chat.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)