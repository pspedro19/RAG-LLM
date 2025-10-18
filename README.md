#  Curacao Tourism Assistant — RAG-LLM Multi-Agent System

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Airflow](https://img.shields.io/badge/Airflow-Orchestration-017CEE?logo=apacheairflow&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-4169E1?logo=postgresql&logoColor=white)
![MinIO](https://img.shields.io/badge/MinIO-Object%20Store-C72E49?logo=minio&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-Semantic%20Search-2C3E50)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Index-2E86C1)
![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **Curacao Tourism Assistant** es un sistema de IA multi-agente que combina **RAG**, **NLP** y **visión por computadora** para ofrecer asistencia turística contextual y trazable sobre la isla de **Curazao**.

---

##  Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Arquitectura y Estructura](#-arquitectura-y-estructura)
- [Instalación](#-instalación)
- [Configuración (.env)](#-configuración-env)
- [Uso Rápido](#-uso-rápido)
- [Módulos Clave](#-módulos-clave)
- [Observabilidad y Seguridad](#-observabilidad-y-seguridad)
- [Despliegue a Producción](#-despliegue-a-producción)
- [Runbook (Comandos Útiles)](#-runbook-comandos-útiles)
- [Testing](#-testing)
- [Solución de Problemas](#-solución-de-problemas)
- [Contribución y Licencia](#-contribución-y-licencia)

---

##  Descripción

- ** Objetivo:** Proveer respuestas confiables, actuales y explicables a turistas y operadores, con trazabilidad desde la fuente.
- ** Enfoque:** Orquestación **multi-agente** (LangGraph), **RAG** con FAISS/pgvector, y **CV** para análisis de imágenes.
- ** Operación:** API **FastAPI**, orquestación **Airflow**, seguimiento de experimentos **MLflow**, almacenamiento de objetos **MinIO**, **PostgreSQL** como store transaccional.

---

##  Características

-  **Multi-Agente** (enrutamiento, debate, verificación de fuentes)
-  **RAG Avanzado** (FAISS/pgvector, chunking, reranking opcional)
-  **Visión por Computadora** (clasificación/identificación básica)
-  **Resiliencia** (circuit-breaker, backoff, fallback entre LLMs)
-  **Trazabilidad** (MLflow, checkpoints, logs estructurados)
-  **API REST** con **FastAPI** (Swagger/Redoc)
-  **Contenerización** completa (**Docker Compose**)
-  **MLOps** (pipelines de ingestión/entrenamiento/monitoreo)

---

##  Arquitectura y Estructura

```
RAG-LLM/
├─ app/                         # App FastAPI + dominio
│  ├─ core/
│  │  ├─ config/                # Config global (pydantic)
│  │  ├─ db/                    # Conexiones (Postgres/MinIO)
│  │  ├─ pipelines/             # Jobs y flujos internos
│  │  ├─ db_service.py
│  │  ├─ faiss_manager.py
│  │  └─ sync_service.py
│  ├─ data/
│  │  ├─ documents/             # PDFs/Docs turísticos
│  │  └─ indices/               # Índices FAISS
│  ├─ embeddings/
│  │  ├─ embedding_service.py
│  │  └─ embedding_repository.py
│  ├─ ingestion/
│  │  ├─ document_processor.py
│  │  ├─ document_reader.py
│  │  └─ chunk_repository.py
│  ├─ query/
│  │  ├─ query_processor.py
│  │  ├─ retriever.py
│  │  └─ context_builder.py
│  ├─ vision/
│  │  ├─ image_processor.py
│  │  ├─ model_service.py
│  │  └─ imagenet_classes.json
│  ├─ notifications/
│  │  ├─ email_service.py
│  │  └─ webhook_service.py
│  ├─ agent_service.py
│  ├─ agent_tp3.py
│  ├─ langgraph_cv_agents_TP2.py
│  ├─ rag_chat_TP1.py
│  ├─ rag_setup.py
│  ├─ main.py                   # FastAPI entrypoint
│  └─ checkpoints/
├─ airflow/
│  ├─ dags/
│  │  ├─ DAG_finetuning.py
│  │  ├─ DAG_finetuning_Qlora.py
│  │  └─ DAG_finetuning_RP.py
│  ├─ secrets/
│  │  ├─ connections.yaml
│  │  └─ variables.yaml
│  └─ requirements.txt
├─ chat-Interface/              # Interfaz Django (opcional)
│  ├─ chat/ (models, views, utils)
│  ├─ chatbot/ (settings, urls)
│  ├─ property_images/
│  └─ templates/
├─ mlflow/ (Dockerfile, requirements.txt)
├─ frontend/ (index.html, app.js, styles.css)
├─ postgres/ (Dockerfile)
├─ minio/ (Dockerfile)
├─ checkpoints/
├─ docker-compose.yaml
└─ requirements.txt
```

**Diagrama (alto nivel):**
```
Usuarios ⇄ FastAPI (app/main.py)
           │
           ├─ Multi-Agente (agent_service.py / agent_tp3.py)
           │    ├─ RAG (retriever.py + FAISS/pgvector)
           │    └─ Visión (vision/model_service.py)
           │
           ├─ Postgres (metadatos) ─── MinIO (objetos)
           ├─ Airflow (DAGs MLOps)
           └─ MLflow (experimentos / métricas)
```

---

##  Instalación

```bash
# 1) Clonar
git clone https://github.com/pspedro19/RAG-LLM.git
cd RAG-LLM

# 2) Entorno virtual
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
# venv\Scripts\activate

# 3) Dependencias
pip install -r requirements.txt
# (si aplican reqs adicionales en app/)
pip install -r app/requirements.txt
```

---

##  Configuración (.env)

Crea tu `.env` desde el ejemplo y edítalo:

```bash
cp .env.example .env
```

```env
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
TAVILY_API_KEY=...

# DB / Storage
POSTGRES_URL=postgresql://user:pass@localhost:5432/curacao_rag

# Security
JWT_SECRET_KEY=_min_32_chars_secret_key
ENCRYPTION_KEY=_32_bytes_base64_key

# Observability
MLFLOW_TRACKING_URI=http://localhost:5000
```

> **Tip:** Asegúrate de que `JWT_SECRET_KEY` y `ENCRYPTION_KEY` sean **largos y seguros**. En producción, usa un **secret manager**.

---

##  Uso Rápido

### Opción A) Docker Compose (Full Stack)

```bash
# Levantar todo
docker-compose up -d --build

# Ver estado
docker-compose ps

# Logs de la app
docker-compose logs -f app
```

- FastAPI Docs: `http://localhost:8000/docs` (Swagger) • `http://localhost:8000/redoc`
- MLflow UI: `http://localhost:5000`
- Airflow UI: `http://localhost:8080`

### Opción B) Local (solo API)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

##  Módulos Clave

###  Multi-Agente (TP3)
```bash
python -m app.agent_tp3
# o vía API
curl -sX POST "http://localhost:8000/agent/query"   -H "Content-Type: application/json"   -d '{"message":"Itinerario 3 días en Curazao (playas + historia)","language":"es"}'
```

### 🔍 RAG Básico (TP1)
```bash
# Preparar datos
python -m app.rag_setup clean
python -m app.rag_setup ingest --doc-dir app/data/documents/
python -m app.rag_setup sync
python -m app.rag_setup rebuild

# Chat RAG
python -m app.rag_chat_TP1 --model openai --top-k 10
```

### 👥 Agentes para CVs (TP2)
```bash
python -m app.langgraph_cv_agents_TP2
curl -sX POST "http://localhost:8000/agents/cv"   -H "Content-Type: application/json"   -d '{"question":"¿Experiencia en IA de Pedro?","person":"pedro"}'
```

---

##  Observabilidad y Seguridad

- **Resiliencia:** circuit-breaker, reintentos con backoff, cache de embeddings, fallback entre OpenAI/Anthropic/local.
- **Seguridad:** JWT en endpoints, cifrado AES-256 para secretos en repos, sanitización de prompts.
- **Métricas/Logs:** MLflow (experimentos, artefactos), logs estructurados (`app/curacao_assistant.log`), health checks.

**Salud de la app:**
```bash
curl http://localhost:8000/health/metrics
python -m app.health_checker
```

---

##  Despliegue a Producción

Variables recomendadas:

```env
ENVIRONMENT=production
LOG_LEVEL=INFO
POSTGRES_URL=postgresql://user:pass@postgres:5432/curacao_rag_prod
JWT_SECRET_KEY=_super_long_secure_key
ENCRYPTION_KEY_32_bytes_prod_key
MLFLOW_TRACKING_URI=http://mlflow:5000
AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql://airflow:airflow@postgres:5432/airflow_prod
```

Comandos:

```bash
# Despliegue (compose + .env.prod)
docker-compose -f docker-compose.yaml --env-file .env.prod up -d

# Backup Postgres
docker-compose exec postgres pg_dump -U user curacao_rag > backup_$(date +%F).sql

# Logs en vivo
docker-compose logs -f --tail=100 app
```

---

##  Runbook (Comandos Útiles)

```bash
# Ingesta RAG
python -m app.rag_setup ingest --doc-dir app/data/documents/

# Reconstruir índice
python -m app.rag_setup rebuild

# Re-sincronizar metadatos
python -m app.rag_setup sync

# Regenerar embeddings (selectivo)
python -m app.embeddings.embedding_service --refresh --only-updated
```

---

##  Testing

```bash
pytest app/test/ -v
pytest app/test/test_embeddings.py -v
pytest app/test/test_integration_local.py -v
pytest app/test/test_e2e.py -v
pytest app/test/test_search.py -v   # rendimiento
```

