"""
Image Processor - Utilidades para procesamiento de imágenes

Proporciona funciones auxiliares para validación, redimensionamiento
y optimización de imágenes antes de la inferencia.
"""

from PIL import Image
import io
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Procesador de imágenes con validación y optimización"""

    # Formatos soportados
    SUPPORTED_FORMATS = {'JPEG', 'PNG', 'JPG', 'BMP', 'WEBP'}

    # Tamaño máximo (10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024

    @staticmethod
    def validate_image(file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Valida que el archivo sea una imagen válida

        Args:
            file_path: Ruta al archivo

        Returns:
            (es_valido, mensaje_error)
        """
        try:
            # Verificar tamaño de archivo
            import os
            file_size = os.path.getsize(file_path)

            if file_size > ImageProcessor.MAX_FILE_SIZE:
                return False, f"Imagen demasiado grande ({file_size / 1024 / 1024:.1f}MB). Máximo: 10MB"

            # Intentar abrir la imagen
            with Image.open(file_path) as img:
                # Verificar formato
                if img.format not in ImageProcessor.SUPPORTED_FORMATS:
                    return False, f"Formato {img.format} no soportado. Formatos válidos: {ImageProcessor.SUPPORTED_FORMATS}"

                # Verificar dimensiones mínimas
                if img.size[0] < 32 or img.size[1] < 32:
                    return False, f"Imagen demasiado pequeña ({img.size}). Mínimo: 32x32"

                # Verificar que no esté corrupta
                img.verify()

            return True, None

        except Exception as e:
            return False, f"Error validando imagen: {str(e)}"

    @staticmethod
    def optimize_image(file_path: str, output_path: str, max_size: Tuple[int, int] = (1024, 1024)) -> bool:
        """
        Optimiza una imagen redimensionándola si es necesario

        Args:
            file_path: Ruta de entrada
            output_path: Ruta de salida
            max_size: Tamaño máximo (ancho, alto)

        Returns:
            True si se optimizó exitosamente
        """
        try:
            with Image.open(file_path) as img:
                # Convertir a RGB si es necesario
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Redimensionar si excede tamaño máximo
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.LANCZOS)
                    logger.info(f"Imagen redimensionada de {img.size} a {max_size}")

                # Guardar optimizada
                img.save(output_path, format='JPEG', quality=85, optimize=True)

            return True

        except Exception as e:
            logger.error(f"Error optimizando imagen: {e}")
            return False

    @staticmethod
    def get_image_metadata(file_path: str) -> dict:
        """
        Extrae metadata de una imagen

        Args:
            file_path: Ruta a la imagen

        Returns:
            Dict con metadata
        """
        try:
            with Image.open(file_path) as img:
                return {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "width": img.size[0],
                    "height": img.size[1],
                    "aspect_ratio": img.size[0] / img.size[1]
                }
        except Exception as e:
            logger.error(f"Error extrayendo metadata: {e}")
            return {}
