"""
Vision Module - Inference con modelos CNN/ViT preentrenados

Este módulo proporciona capacidades de visión por computadora
para el proyecto RAG-LLM, cumpliendo con el requisito de
"integración de modelo preexistente para inferencia".
"""

from .model_service import VisionModelService
from .image_processor import ImageProcessor

__all__ = ['VisionModelService', 'ImageProcessor']
