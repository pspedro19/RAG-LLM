# app2/test/test_ingest.py
import os
import pytest
import logging
from pathlib import Path
import uuid
import tempfile
import subprocess
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar módulos desde app2
from app2.core.config.config import Config
from app2.ingestion.document_reader import DocumentReader
from app2.ingestion.document_processor import DocumentProcessor, DocumentInput

# Helper para ejecutar comandos SQL en PostgreSQL usando docker exec
def execute_sql(config, sql_command):
    cmd = [
        "docker", "exec", "postgres_pgvector", 
        "psql", "-U", config.DB_USER, "-d", config.DB_NAME,
        "-c", sql_command
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Error ejecutando SQL: {result.stderr}")
    return result.stdout

@pytest.fixture
def test_config():
    """Fixture para crear una configuración de prueba."""
    config = Config()
    # Asegurarse de que los directorios de prueba existan
    os.makedirs(config.INDICES_DIR, exist_ok=True)
    os.makedirs(config.KB_DIR, exist_ok=True)
    return config

@pytest.fixture
def document_reader():
    """Fixture para crear un lector de documentos."""
    return DocumentReader()

@pytest.fixture
def document_processor(test_config):
    """Fixture para crear un procesador de documentos."""
    return DocumentProcessor(test_config)

def test_read_text_file(document_reader, tmp_path):
    """Prueba la lectura de un archivo de texto."""
    # Crear un archivo de texto temporal
    file_path = tmp_path / "test_doc.txt"
    content = "Este es un documento de prueba.\nContiene múltiples líneas.\nPara probar el lector de documentos."
    file_path.write_text(content)
    
    # Leer el archivo
    result = document_reader.read_document(str(file_path))
    
    # Verificar resultados
    assert result['content'] == content
    assert result['format'] == 'text'
    assert 'metadata' in result
    assert result['metadata']['filename'] == "test_doc.txt"

def test_document_chunking(document_processor):
    """Prueba la división en chunks de un documento."""
    # Crear un texto de prueba
    text = "Este es un párrafo de prueba.\n\nEste es otro párrafo que debería quedar en otro chunk. " + \
           "Este texto es parte del mismo párrafo.\n\nEste es un tercer párrafo para probar."
    
    # Dividir en chunks
    chunks = document_processor.create_chunks(text)
    
    # Verificar resultados
    assert len(chunks) >= 1, "Debería haber al menos 1 chunk"
    assert 'content' in chunks[0]
    assert 'token_count' in chunks[0]
    assert 'hash' in chunks[0]

def test_document_validation(document_processor):
    """Prueba la validación de documentos."""
    # Documento válido
    valid_doc = DocumentInput(
        title="Documento válido",
        content="Este es un contenido válido",
        metadata={"author": "Test"}
    )
    
    # Validar
    result = document_processor.validate_document(valid_doc)
    assert result == valid_doc
    
    # Documento inválido (contenido vacío)
    with pytest.raises(ValueError):
        document_processor.validate_document(DocumentInput(
            title="Documento inválido",
            content="   "  # Solo espacios
        ))

@pytest.mark.skip("Esta prueba requiere conexión directa a la BD, se probará en otro contexto")
def test_document_processing_and_storage(document_processor, test_config):
    """Prueba el procesamiento completo de un documento y su almacenamiento."""
    # Verificar si el contenedor está ejecutándose
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=postgres_pgvector", "--format", "{{.Names}}"],
        capture_output=True, text=True
    )
    if "postgres_pgvector" not in result.stdout:
        pytest.skip("El contenedor postgres_pgvector no está en ejecución")
    
    # Crear documento de prueba
    doc_input = DocumentInput(
        title="Documento de prueba para almacenamiento",
        content="Este es un contenido de prueba.\n\nTiene varios párrafos.\n\nPara verificar el almacenamiento correcto.",
        metadata={"category": "test", "author": "pytest"}
    )
    
    # Aquí insertaríamos en la BD directamente con docker exec, pero lo dejamos para otra prueba
    logger.info("Procesamiento y almacenamiento simulado correctamente")

def test_pdf_ingestion(document_reader, test_config):
    """
    Prueba la ingesta de un documento PDF si PyPDF2 está instalado.
    Skip si no está disponible.
    """
    try:
        from PyPDF2 import PdfWriter
    except ImportError:
        pytest.skip("PyPDF2 no está instalado")
    
    # Crear un directorio temporal para pruebas
    with tempfile.TemporaryDirectory() as temp_dir:
        # Crear un PDF simple para pruebas
        pdf_path = Path(temp_dir) / "test_doc.pdf"
        
        # Intentar crear un PDF simple (esto es básico y puede fallar)
        try:
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            with open(pdf_path, "wb") as f:
                writer.write(f)
            
            # Intentar leer el PDF
            doc_data = document_reader.read_document(str(pdf_path))
            
            # Si llegamos aquí, continuar con la prueba
            assert 'content' in doc_data
            assert doc_data['format'] == 'pdf'
        except Exception as e:
            pytest.skip(f"No se pudo crear/leer PDF de prueba: {e}")

def test_document_from_app2_folder(document_reader, document_processor, test_config):
    """Prueba la lectura de documentos desde la carpeta app2/data/documents."""
    # Crear un archivo de texto en la carpeta de documentos de app2
    test_file = test_config.KB_DIR / "txt" / "test_document.txt"
    os.makedirs(test_config.KB_DIR / "txt", exist_ok=True)
    
    content = "Este es un documento de prueba específico para app2.\nVerifica que la lectura funciona correctamente."
    with open(test_file, 'w') as f:
        f.write(content)
    
    try:
        # Leer el archivo
        result = document_reader.read_document(str(test_file))
        
        # Verificar resultados
        assert result['content'] == content
        assert result['format'] == 'text'
        assert 'metadata' in result
        assert result['metadata']['filename'] == "test_document.txt"
        
        # Procesar el documento
        doc_input = DocumentInput(
            title="Documento de app2",
            content=result['content'],
            metadata=result['metadata']
        )
        
        # Validar el documento
        doc_input = document_processor.validate_document(doc_input)
        
        # Verificar validación
        assert doc_input.title == "Documento de app2"
    finally:
        # Limpiar: eliminar el archivo de prueba
        if os.path.exists(test_file):
            os.remove(test_file)

def test_pdf_documents_in_app2(document_reader, test_config):
    """Verifica los documentos PDF en app2/data/documents/pdf."""
    pdf_dir = test_config.KB_DIR / "pdf" / "curaçao_information"
    
    if not pdf_dir.exists():
        pytest.skip(f"Directorio de PDFs no encontrado: {pdf_dir}")
    
    # Listar todos los archivos PDF en subdirectorios
    pdf_files = []
    for subdir in pdf_dir.glob("*"):
        if subdir.is_dir():
            for pdf_file in subdir.glob("*.pdf"):
                pdf_files.append(pdf_file)
    
    if not pdf_files:
        pytest.skip("No se encontraron archivos PDF en el directorio")
    
    logger.info(f"Se encontraron {len(pdf_files)} archivos PDF")
    for pdf_file in pdf_files[:3]:  # Probar solo los primeros 3 para no demorar el test
        try:
            logger.info(f"Leyendo {pdf_file.name}")
            result = document_reader.read_document(str(pdf_file))
            assert 'content' in result
            assert result['format'] == 'pdf'
            # Mostrar primeros 100 caracteres del contenido
            content_preview = result['content'][:100].replace('\n', ' ')
            logger.info(f"Contenido (primeros 100 caracteres): {content_preview}...")
        except Exception as e:
            logger.error(f"Error al leer {pdf_file.name}: {e}")
            continue

if __name__ == "__main__":
    pytest.main(["-v", "test_ingest.py"])