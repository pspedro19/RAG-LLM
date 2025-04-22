# app2/core/faiss_manager.py
import faiss
import numpy as np
import os
import logging
from typing import Tuple, List, Dict, Any, Optional
import asyncio

from app2.core.config.config import Config

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    Gestor de índices FAISS para búsqueda de similitud vectorial.
    Proporciona operaciones atómicas sobre índices FAISS.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Inicializa el gestor de FAISS con la configuración proporcionada."""
        self.config = config or Config()
        self.dimension = self.config.VECTOR_SIZE
        self.nprobe = self.config.NPROBE
        
        logger.info(f"Initializing FAISSVectorStore with dimension {self.dimension}")
        
        # Por defecto, usamos un índice plano L2 para máxima precisión
        self.index = faiss.IndexFlatL2(self.dimension)
        
        if not self.index:
            raise RuntimeError("Failed to create FAISS index")
        
        logger.info("FAISS index created successfully")
        
        # Determinar ruta del índice
        self.indices_dir = self.config.INDICES_DIR
        self.backup_dir = self.config.BACKUP_DIR
        
        self.index_path = os.path.join(
            self.indices_dir, 
            f"faiss_index_{self.dimension}.bin"
        )
        
        # Cargar índice existente
        self._load_existing_index()

    def _load_existing_index(self):
        """
        Carga un índice existente desde disco si está disponible.
        Si falla, mantiene el índice vacío creado en el constructor.
        """
        if os.path.exists(self.index_path):
            try:
                logger.info(f"Loading FAISS index from {self.index_path}")
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Index loaded successfully: {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading existing index: {e}")
                logger.info("Using new empty index instead")

    def verify_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Verifica y formatea un vector para operaciones FAISS.
        
        Args:
            vector: Vector a verificar, como numpy array o lista
            
        Returns:
            Vector correctamente formateado como numpy array
            
        Raises:
            ValueError: Si el vector tiene dimensiones incorrectas o valores no finitos
        """
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)
        
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        
        if vector.shape[1] != self.dimension:
            raise ValueError(f"Vector has incorrect dimension: {vector.shape[1]} != {self.dimension}")
        
        if not np.all(np.isfinite(vector)):
            raise ValueError("Vector contains non-finite values (NaN or inf)")
        
        # Asegurar que el array es C-contiguous para mejor rendimiento con FAISS
        vector = np.ascontiguousarray(vector)
        return vector

    def add_vectors(self, vectors: np.ndarray) -> List[int]:
        """
        Añade vectores al índice FAISS y guarda los cambios.
        
        Args:
            vectors: Matriz de vectores a añadir (n_vectors, dimension)
            
        Returns:
            Lista de IDs asignados en el índice (rango consecutivo)
            
        Raises:
            ValueError: Si los vectores tienen formato incorrecto
            RuntimeError: Si falla la operación de añadir
        """
        try:
            vectors = self.verify_vector(vectors)
            logger.info(f"Adding {len(vectors)} vectors to FAISS index")
            
            start_id = self.index.ntotal
            self.index.add(vectors)
            end_id = self.index.ntotal
            
            logger.info(f"Vectors added. IDs range: {start_id} to {end_id-1}")
            
            # Guardar el índice
            self.save_index()
            
            return list(range(start_id, end_id))
        except Exception as e:
            logger.error(f"Error adding vectors to FAISS: {e}")
            raise

    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Busca los k vectores más cercanos al vector de consulta.
        
        Args:
            query_vector: Vector de consulta
            k: Número de resultados a devolver
            
        Returns:
            Tupla de (distancias, ids) donde cada uno es un numpy array
            
        Raises:
            ValueError: Si el vector de consulta tiene formato incorrecto
            RuntimeError: Si falla la operación de búsqueda
        """
        try:
            query_vector = self.verify_vector(query_vector)
            
            # Verificar que el vector de consulta tenga la forma esperada
            if query_vector.shape != (1, self.dimension):
                raise ValueError(f"Query vector shape is {query_vector.shape}, expected (1, {self.dimension})")
                
            logger.info(f"Searching for {k} nearest neighbors")
            
            if self.index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return np.array([]), np.array([])
            
            # Configurar nprobe para índices IVF si aplica
            if hasattr(self.index, 'nprobe'):
                original_nprobe = self.index.nprobe
                self.index.nprobe = self.nprobe
            
            # Buscar vectores similares
            distances, ids = self.index.search(query_vector, min(k, self.index.ntotal))
            
            # Restaurar nprobe original si fue modificado
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = original_nprobe
                
            logger.info(f"Search completed: {len(ids[0])} results")
            return distances, ids
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}")
            raise

    def get_index_info(self) -> Dict[str, Any]:
        """
        Retorna información sobre el estado actual del índice FAISS.
        
        Returns:
            Diccionario con metadatos del índice
        """
        info = {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': type(self.index).__name__,
            'is_trained': getattr(self.index, 'is_trained', True),
            'status': 'ready'
        }
        
        # Añadir información específica de índices IVF si aplica
        if hasattr(self.index, 'nlist'):
            info['nlist'] = self.index.nlist
            info['nprobe'] = getattr(self.index, 'nprobe', self.nprobe)
        
        return info

    def reset_index(self):
        """
        Reinicializa el índice FAISS a su estado inicial (vacío) y elimina el archivo persistente.
        """
        if os.path.exists(self.index_path):
            try:
                os.remove(self.index_path)
                logger.info(f"Persistent index file removed: {self.index_path}")
            except Exception as e:
                logger.error(f"Error removing persistent index: {e}")
        
        self.index = faiss.IndexFlatL2(self.dimension)
        logger.info("FAISS index reset to empty state")

    def save_index(self) -> bool:
        """
        Guarda el índice actual en disco.
        
        Returns:
            bool: True si el guardado fue exitoso, False en caso contrario
        """
        try:
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Guardar el índice
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Index saved to {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False

    def backup_index(self) -> Optional[str]:
        """
        Crea una copia de respaldo del índice actual.
        
        Returns:
            Ruta al archivo de respaldo o None si falla
        """
        try:
            # Crear directorio de respaldos si no existe
            os.makedirs(self.backup_dir, exist_ok=True)
            
            # Generar nombre para el respaldo con timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(
                self.backup_dir, 
                f"faiss_index_{self.dimension}_{timestamp}.bin"
            )
            
            # Guardar copia
            faiss.write_index(self.index, backup_path)
            logger.info(f"Index backup created at {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create index backup: {e}")
            return None