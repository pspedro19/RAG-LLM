#!/usr/bin/env python3
"""
Sistema de Agentes Conversacional para Consulta de CVs con LangGraph

Este script implementa un sistema de agentes conversacional basado en LangGraph 
para consultar los CVs de diferentes personas (Pedro, Jorge, Leonardo) de manera eficiente.

Características:
1. Un agente por persona
2. Detección inteligente de menciones a personas en la consulta
3. Procesamiento paralelo para consultas comparativas
4. Agente por defecto cuando no se menciona a nadie
5. Respuestas conversacionales generadas por OpenAI
6. Mantenimiento de historial de conversación

Uso:
    python langgraph_cv_agents.py
"""

import asyncio
import re
import os
import sys
import logging
from typing import Dict, List, TypedDict, Any, Tuple
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI

# Asegurar que se puede importar cv_assistant
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar la clase EnhancedCVAssistant
from rag_chat_TP1 import EnhancedCVAssistant, Colors

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("langgraph_cv_agents.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("langgraph-cv-agents")

# Inicializar cliente de OpenAI
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 1. Definición del estado del grafo (actualizado para conversación)
class AgentState(TypedDict):
    query: str                      # Consulta del usuario
    people: List[str]               # Personas detectadas en la consulta
    contexts: Dict[str, str]        # Contextos recuperados por persona
    raw_response: str               # Respuesta RAG sin procesar
    conversation_history: List[Dict[str, str]]  # Historial de la conversación
    final_response: str             # Respuesta final generada por OpenAI

# Historial de conversación global
CONVERSATION_HISTORY = []

# Inicializar una única instancia del EnhancedCVAssistant para reutilizar la conexión a la BD
_assistant = None

async def get_assistant():
    """Obtiene o inicializa la instancia única de EnhancedCVAssistant."""
    global _assistant
    if _assistant is None:
        _assistant = EnhancedCVAssistant(top_k=5, use_localhost=True)
        await _assistant.initialize()
    return _assistant

# Función para sanitizar texto y prevenir errores de codificación
def sanitize_text(text):
    """
    Sanitiza el texto para eliminar caracteres problemáticos
    o surrogates que puedan causar errores de codificación.
    
    Args:
        text: Texto a sanitizar
        
    Returns:
        Texto sanitizado
    """
    if isinstance(text, str):
        # Reemplaza caracteres inválidos con espacios
        return text.encode('utf-8', errors='replace').decode('utf-8')
    return text

# 2. Función condicional para detectar personas en la consulta (Conditional Edge)
def detect_people(state: AgentState) -> str:
    """
    Detecta qué personas se mencionan en la consulta.
    
    Args:
        state: Estado actual con la consulta del usuario
        
    Returns:
        "single" si se detecta una sola persona o ninguna
        "parallel" si se detectan múltiples personas
    """
    query = state["query"].lower()
    people = []
    
    # Patrones mejorados para detectar menciones con variantes de nombres
    patterns = {
        'pedro': r'\b(pedro|pérez|pedrito)\b',
        'jorge': r'\b(jorge|hern[aá]n|cuenca|mar[ií]n)\b',
        'leonardo': r'\b(leonardo|leo|ortiz|arismendi)\b'
    }
    
    # Verificar coincidencias para cada persona
    for person, pattern in patterns.items():
        if re.search(pattern, query, re.IGNORECASE):
            people.append(person)
    
    # Detectar consultas comparativas o que impliquen a todos
    if re.search(r'\b(todos|los tres|los 3|compar(a|ar)|vs|versus)\b', query, re.IGNORECASE):
        people = ['pedro', 'jorge', 'leonardo']
    
    # Si no se menciona a nadie, usar el agente por defecto (estudiante)
    state["people"] = people if people else ['default']
    
    logger.info(f"Personas detectadas en la consulta: {state['people']}")
    
    # Determinar ruta: procesamiento paralelo o individual
    return "parallel" if len(state["people"]) > 1 else "single"

# 3. Implementación de los agentes específicos por persona
async def pedro_agent(state: AgentState) -> AgentState:
    """Agente que procesa consultas sobre Pedro."""
    logger.info("Procesando consulta con el agente de Pedro")
    return await process_agent('pedro', state)

async def jorge_agent(state: AgentState) -> AgentState:
    """Agente que procesa consultas sobre Jorge."""
    logger.info("Procesando consulta con el agente de Jorge")
    return await process_agent('jorge', state)

async def leonardo_agent(state: AgentState) -> AgentState:
    """Agente que procesa consultas sobre Leonardo."""
    logger.info("Procesando consulta con el agente de Leonardo")
    return await process_agent('leonardo', state)

async def default_agent(state: AgentState) -> AgentState:
    """Agente por defecto (estudiante) cuando no se especifica persona."""
    logger.info("Procesando consulta con el agente por defecto")
    # Asumimos que el estudiante es 'pedro', puedes cambiarlo según corresponda
    return await process_agent('pedro', state)

# 4. Función auxiliar para procesar consultas con el RAG existente
async def process_agent(person: str, state: AgentState) -> AgentState:
    """
    Procesa una consulta para una persona específica usando el sistema RAG.
    
    Args:
        person: Identificador de la persona ('pedro', 'jorge', 'leonardo', 'default')
        state: Estado actual del grafo
        
    Returns:
        Estado actualizado con el contexto recuperado
    """
    # Obtener la instancia compartida del asistente
    assistant = await get_assistant()
    
    # Obtener los doc_ids para la persona especificada
    doc_ids = assistant.person_doc_ids.get(person, [])
    
    # Si no hay doc_ids para esta persona, registrar el error
    if not doc_ids:
        logger.warning(f"No se encontraron documentos para {person}")
        state["contexts"][person] = f"No se encontraron documentos para {person}."
        return state
    
    # Sanitizar la consulta antes de la búsqueda
    sanitized_query = sanitize_text(state["query"])
    
    # Realizar búsqueda filtrada con el sistema RAG existente
    logger.info(f"Realizando búsqueda para {person} con doc_ids: {doc_ids}")
    results = await assistant._perform_filtered_search(sanitized_query, doc_ids)
    
    # Guardar el contexto recuperado en el estado
    if results and results.get('context'):
        state["contexts"][person] = results.get('context', '')
        logger.info(f"Contexto recuperado para {person} ({len(state['contexts'][person])} caracteres)")
    else:
        logger.warning(f"No se encontró información relevante para {person}")
        state["contexts"][person] = f"No se encontró información relevante para {person} sobre: {state['query']}"
    
    return state

# 5. Función para procesar múltiples agentes en paralelo
async def process_multiple_agents(state: AgentState) -> AgentState:
    """
    Ejecuta múltiples agentes en paralelo para consultas comparativas.
    
    Args:
        state: Estado actual con las personas a consultar
        
    Returns:
        Estado actualizado con los contextos de todas las personas
    """
    logger.info(f"Procesando en paralelo para personas: {state['people']}")
    tasks = []
    
    # Crear una tarea asíncrona para cada persona en la consulta
    for person in state["people"]:
        if person == "pedro":
            tasks.append(asyncio.create_task(pedro_agent(state)))
        elif person == "jorge":
            tasks.append(asyncio.create_task(jorge_agent(state)))
        elif person == "leonardo":
            tasks.append(asyncio.create_task(leonardo_agent(state)))
        elif person == "default":
            tasks.append(asyncio.create_task(default_agent(state)))
    
    # Esperar a que todas las tareas terminen
    await asyncio.gather(*tasks)
    logger.info(f"Procesamiento paralelo completado, recuperados {len(state['contexts'])} contextos")
    return state

# 6. Nodo para combinar resultados y formatear la respuesta RAG
def combine_results(state: AgentState) -> AgentState:
    """
    Combina los resultados de múltiples agentes en una respuesta RAG.
    
    Args:
        state: Estado con los contextos recuperados
        
    Returns:
        Estado actualizado con la respuesta RAG
    """
    contexts = state["contexts"]
    
    # Si es una consulta comparativa (múltiples personas)
    if len(contexts) > 1:
        logger.info("Generando respuesta comparativa RAG")
        # Crear una respuesta estructurada comparando perfiles
        response = ""
        
        # Nombres completos para mejor presentación
        full_names = {
            'pedro': "Pedro Pérez",
            'jorge': "Jorge Hernán Cuenca Marín",
            'leonardo': "Leonardo Ortiz Arismendi",
            'default': "Estudiante (Perfil por defecto)"
        }
        
        # Para cada persona, agregar su sección
        for person in state["people"]:
            if person in contexts and contexts[person]:
                full_name = full_names.get(person, person.capitalize())
                
                # Añadir encabezado para cada persona
                response += f"\n\n[Información sobre {full_name}]\n\n"
                response += contexts[person]
    else:
        # Para consulta individual, usar directamente el contexto
        logger.info("Generando respuesta individual RAG")
        person = state["people"][0] if state["people"] else "default"
        context = contexts.get(person, "No se encontró información.")
        response = context
    
    # Guardar la respuesta RAG en el estado
    state["raw_response"] = response
    
    # Inicializar historial de conversación si no existe
    if "conversation_history" not in state:
        state["conversation_history"] = []
    
    return state

# 7. Nuevo nodo para generar respuesta conversacional con OpenAI
async def llm_generate_response(state: AgentState) -> AgentState:
    """
    Genera una respuesta conversacional usando OpenAI basada en el contexto RAG
    
    Args:
        state: Estado con el contexto RAG
        
    Returns:
        Estado actualizado con la respuesta final generada
    """
    # Prepara la historia de conversación para el prompt
    conversation_history = state.get("conversation_history", [])
    
    # Construye el prompt con contexto y conversación
    system_prompt = """Eres un asistente experto en información curricular que proporciona información precisa y útil.
    Responde basándote ÚNICAMENTE en el CONTEXTO proporcionado. Si no tienes suficiente información en el CONTEXTO,
    indícalo claramente. Mantén un tono profesional y amigable.
    
    La información que proporcionas debe ser específica, relevante y directamente basada en los CVs. No inventes
    información que no esté en el contexto. Si te preguntan por algo que no está en el contexto, dilo claramente.
    
    Si la consulta compara a varias personas, asegúrate de estructurar tu respuesta para mostrar claramente
    las similitudes y diferencias relevantes.
    """
    
    # Crea los mensajes para la conversación
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    # Añade historial de conversación si existe
    if conversation_history:
        # Incluir solo las últimas 5 interacciones para mantener el contexto sin exceder tokens
        for message in conversation_history[-5:]:
            messages.append(message)
    
    # Añade el contexto recuperado
    context = f"CONTEXTO:\n{state['raw_response']}"
    
    # Determinar si es consulta comparativa para ajustar las instrucciones
    if len(state["people"]) > 1:
        query_instruction = f"El usuario está preguntando sobre una comparación entre {', '.join(state['people'])}. " + \
                           f"La consulta es: {state['query']}\n\n{context}"
    else:
        person = state["people"][0] if state["people"] else "default"
        query_instruction = f"El usuario está preguntando sobre {person}. " + \
                          f"La consulta es: {state['query']}\n\n{context}"
    
    messages.append({"role": "user", "content": query_instruction})
    
    try:
        # Llamada a OpenAI
        logger.info("Enviando solicitud a OpenAI para generar respuesta conversacional")
        response = await client.chat.completions.create(
            model="gpt-4-turbo",  # O el modelo que prefieras
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Obtener la respuesta
        response_text = response.choices[0].message.content
        logger.info(f"Respuesta conversacional generada ({len(response_text)} caracteres)")
        
        # Actualizar el estado
        state["final_response"] = response_text
        
        # Actualizar el historial de conversación
        state["conversation_history"].append({"role": "user", "content": state["query"]})
        state["conversation_history"].append({"role": "assistant", "content": response_text})
        
    except Exception as e:
        logger.error(f"Error generando respuesta con OpenAI: {e}")
        # Fallback: usar la respuesta RAG directamente
        state["final_response"] = f"Lo siento, tuve un problema generando una respuesta natural. Aquí está la información disponible:\n\n{state['raw_response']}"
    
    return state

# 8. Función para construir y compilar el grafo (actualizado para conversación)
def build_graph() -> StateGraph:
    """
    Construye el grafo de LangGraph para el sistema de agentes conversacional.
    
    Returns:
        Grafo compilado listo para ser invocado
    """
    logger.info("Construyendo grafo de agentes conversacional")
    builder = StateGraph(AgentState)
    
    # Agregar nodos
    builder.add_node("pedro", pedro_agent)
    builder.add_node("jorge", jorge_agent)
    builder.add_node("leonardo", leonardo_agent)
    builder.add_node("default", default_agent)
    builder.add_node("process_multiple", process_multiple_agents)
    builder.add_node("combine", combine_results)
    builder.add_node("generate", llm_generate_response)  # Nuevo nodo para generación OpenAI
    
    # Define un enrutador mejorado que decide y ejecuta el agente apropiado
    async def enhanced_router(state: AgentState):
        logger.info("Enrutador procesando consulta")
        # Detectar personas en la consulta
        result = detect_people(state)
        
        # Procesar según el resultado de la detección
        if result == "single":
            person = state["people"][0]
            logger.info(f"Enrutando a agente individual: {person}")
            
            # Llamar al agente apropiado directamente
            if person == "pedro":
                return await pedro_agent(state)
            elif person == "jorge":
                return await jorge_agent(state)
            elif person == "leonardo":
                return await leonardo_agent(state)
            else:  # default
                return await default_agent(state)
        else:  # parallel
            logger.info("Enrutando a procesamiento múltiple")
            return await process_multiple_agents(state)
    
    # Agregar el enrutador mejorado
    builder.add_node("enhanced_router", enhanced_router)
    
    # Establecer punto de entrada
    builder.set_entry_point("enhanced_router")
    
    # Configurar flujo: router -> combine -> generate -> END
    builder.add_edge("enhanced_router", "combine")
    builder.add_edge("combine", "generate")
    builder.add_edge("generate", END)
    
    # Compilar el grafo
    return builder.compile()

# 9. Variable global para el grafo compilado (se inicializa una vez)
GRAPH = None

# 10. Función principal para procesar consultas (actualizada para conversación)
async def answer(query: str) -> str:
    """
    Procesa una consulta a través del grafo de agentes y mantiene estado de conversación.
    
    Args:
        query: Consulta del usuario
        
    Returns:
        Respuesta conversacional generada
    """
    global GRAPH, CONVERSATION_HISTORY
    
    # Sanitizar la consulta
    sanitized_query = sanitize_text(query)
    
    # Inicializar el grafo si no existe
    if GRAPH is None:
        GRAPH = build_graph()
    
    logger.info(f"Procesando consulta conversacional: '{sanitized_query}'")
    
    # Ejecutar el grafo con la consulta sanitizada y el historial
    result = await GRAPH.ainvoke({
        "query": sanitized_query,
        "people": [],
        "contexts": {},
        "raw_response": "",
        "conversation_history": CONVERSATION_HISTORY,
        "final_response": ""
    })
    
    # Actualizar el historial global
    CONVERSATION_HISTORY = result["conversation_history"]
    
    logger.info("Consulta conversacional procesada exitosamente")
    return result["final_response"]

# 11. Función para la interfaz de línea de comandos (actualizada)
async def run_cv_assistant():
    """
    Función principal para ejecutar el asistente conversacional interactivo.
    """
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}{Colors.GREEN}Asistente Conversacional de CVs{Colors.RESET}".center(60))
    print("=" * 60 + "\n")
    
    print("Puedes preguntar sobre Pedro, Jorge, Leonardo o comparar sus perfiles.")
    print("Este asistente es conversacional, así que puedes hacer preguntas de seguimiento.")
    print("Ejemplos de consultas:")
    print(" - ¿Qué experiencia tiene Jorge?")
    print(" - ¿Cuáles son sus estudios?")
    print(" - Compara las habilidades de Pedro y Leonardo")
    print(" - ¿Qué formación tienen los tres?")
    print("\nEscribe 'salir' para terminar.\n")
    
    # Inicializar el asistente y el grafo
    await get_assistant()
    
    try:
        while True:
            # Solicitar entrada
            query = input(f"\n{Colors.BOLD}{Colors.GREEN}Tú:{Colors.RESET} ")
            
            # Verificar si se desea salir
            if query.lower() in ['salir', 'exit', 'quit']:
                print(f"\n{Colors.GREEN}¡Hasta pronto!{Colors.RESET}\n")
                break
                
            if not query.strip():
                continue
                
            # Mostrar estado de procesamiento
            print(f"\n{Colors.DIM}Procesando...{Colors.RESET}", flush=True)
            
            # Sanitizar la entrada
            sanitized_query = sanitize_text(query)
            
            # Procesar consulta conservando contexto conversacional
            response = await answer(sanitized_query)
            
            # Mostrar respuesta
            print(f"\n{Colors.BOLD}{Colors.BLUE}Asistente:{Colors.RESET}")
            print(response)
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.GREEN}¡Hasta pronto!{Colors.RESET}\n")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n{Colors.RED}Error inesperado: {e}{Colors.RESET}\n")
    finally:
        # Cerrar recursos
        if _assistant:
            await _assistant.close()

# Para ejecutar directamente el script
if __name__ == "__main__":
    try:
        asyncio.run(run_cv_assistant())
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)