# RAG-LLM: Guía Completa

## Estructura del Proyecto RAG-LLM

Basado en la estructura de archivos proporcionada, este proyecto está organizado como una plataforma robusta para el desarrollo de aplicaciones de IA centradas en RAG (Retrieval-Augmented Generation) y sistemas de agentes:

```
RAG-LLM/
├── app/                  # Directorio principal de la aplicación
│   ├── core/             # Funcionalidades centrales
│   ├── data/             # Datos de la aplicación
│   │   ├── documents/pdf/
│   │   │   ├── information/  # Documentos para TP1 y TP2
│   │   │   └── tp3_agent/    # Documentos para TP3
│   │   └── indices/      # Índices FAISS para búsqueda vectorial
│   ├── embeddings/       # Servicios de embeddings
│   ├── ingestion/        # Procesamiento y chunking de documentos
│   ├── query/            # Procesamiento de consultas y recuperación
│   ├── rag_chat_TP1.py   # Implementación de chatbot RAG (TP1)
│   ├── langgraph_cv_agents_TP2.py # Sistema de agentes para CVs (TP2)
│   ├── agent_tp3.py      # Sistema multi-agente con razonamiento (TP3)
│   └── rag_setup.py      # Utilidad para configurar datos RAG
├── airflow/              # Configuración y DAGs de Airflow para orquestación
├── chat-Interface/       # Interfaz web basada en Django
├── checkpoints/          # Archivos de checkpoints para los modelos
├── docker-compose.yaml   # Configuración para despliegue con Docker
└── requirements.txt      # Dependencias del proyecto
```

## Guía de Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone [https://github.com/tu-usuario/RAG-LLM.git](https://github.com/pspedro19/RAG-LLM.git)
cd RAG-LLM
```

### 2. Crear y activar entorno virtual
```bash
sudo apt update
```

```bash
sudo apt install -y python3 python3-venv python3-pip
```

```
sudo apt install -y python-is-python3
```

```bash
cd RAG-LLM
```

```bash
python -m venv venv
```
# En Windows
```
venv\Scripts\activate
```
# En Linux/Mac
```
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Crear un archivo `.env` a partir del archivo ejemplo:

```bash
cp example.env .env
nano .env  # Editar con tus claves API
```

Asegúrate de configurar las siguientes variables:

```
TAVILY_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

### 5. Preparar documentos

Asegúrate de que los documentos estén en los directorios correctos:

```
app/data/documents/pdf/information/  # Para TP1 y TP2
app/data/documents/pdf/tp3_agent/    # Para TP3
```


### 6. Montar Microservicio

Asegúrate de que los documentos estén en los directorios correctos:

```
apt install docker-compose
```

```
docker-compose build
```

```
docker-compose up -d
```

## Implementación de los Trabajos Prácticos

### TP1: Chatbot RAG

**Consigna**: Implementar un sistema de generación de texto (chatbot) que utilice la técnica de Retrieval-Augmented Generation (RAG). El chatbot recuperará información de una base de datos (documentos) y la usará para generar respuestas completas.

#### Preparación de datos para TP1

```bash
# Limpiar datos existentes (si es necesario)
python -m app.rag_setup clean

# Ingestar documentos para el sistema RAG
python -m app.rag_setup ingest --doc-dir app/data/documents/pdf/information

# Sincronizar datos
python -m app.rag_setup sync

# Reconstruir índice
python -m app.rag_setup rebuild
```

#### Ejecución del chatbot RAG (TP1)

```bash
# Iniciar la aplicación RAG
python -m app.rag_chat_TP1 --model openai --top-k 16
```

#### Funcionalidades demostradas en TP1
* Carga y procesamiento de documentos PDF
* Generación de embeddings para búsqueda semántica
* Consultas sobre la información contenida en los documentos
* Respuestas generadas aumentadas con información recuperada

### TP2: Sistema de Agentes para CVs

**Consigna**: Implementar un sistema de agentes que responda eficientemente dependiendo de qué persona se está preguntando (1 agente por persona). Por defecto, cuando no se nombra a nadie, utilizar el agente del alumno.

#### Preparación de datos para TP2

```bash
# Utiliza los mismos datos preparados para TP1
# O si necesitas recargar:

# Limpiar datos existentes (si es necesario)
python -m app.rag_setup clean

# Ingestar documentos para el sistema de agentes
python -m app.rag_setup ingest --doc-dir app/data/documents/pdf/information

# Sincronizar datos
python -m app.rag_setup sync

# Reconstruir índice
python -m app.rag_setup rebuild
```

#### Ejecución del sistema de agentes (TP2)

```bash
# Iniciar la aplicación con langgraph para el manejo de agentes
python -m app.langgraph_cv_agents_TP2
```

#### Funcionalidades demostradas en TP2
* Sistema multi-agente con un agente específico para cada miembro del equipo
* Identificación automática del agente correcto basado en consultas
* Respuestas contextualizadas según el CV consultado
* Manejo de consultas que involucran múltiples CVs
* Comportamiento por defecto cuando no se especifica persona

### TP3: LLM con Razonamiento Multi-Agente

**Consigna**: Implementar una aplicación que funcione como un LLM con razonamiento, que reciba una pregunta compleja y utilice diferentes agentes para resolver parcialmente y luego compaginar todas las respuestas para ofrecer la solución.

#### Preparación de datos para TP3

```bash
# Limpiar datos existentes (si es necesario)
python -m app.rag_setup clean

# Ingestar documentos para agentes especializados
python -m app.rag_setup ingest --doc-dir app/data/documents/pdf/tp3_agent

# Sincronizar datos
python -m app.rag_setup sync

# Reconstruir índice
python -m app.rag_setup rebuild
```

#### Ejecución del sistema de razonamiento (TP3)

```bash
# Iniciar la aplicación multi-agente con razonamiento
python -m app.agent_tp3
```

#### Funcionalidades demostradas en TP3
* Procesamiento de preguntas complejas
* Descomposición en sub-problemas asignados a agentes especializados
* Razonamiento intermedio y resolución parcial por agentes
* Compaginación de respuestas parciales en una solución completa
* Contabilización y visualización de tokens (entrada, salida, razonamiento)
