"""
Vision Model Service - Inferencia con CNN/ViT

Proporciona clasificación de imágenes usando modelos preentrenados:
- Vision Transformer (ViT-B/16)
- EfficientNet-B0
- ResNet50

Requisito del proyecto: "Integración de modelo preexistente para inferencia"
"""

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional
import time
import json
import os

logger = logging.getLogger(__name__)


class VisionModelService:
    """Servicio de inferencia con modelos de visión preentrenados"""

    # Mapeo de ImageNet classes (top 1000)
    IMAGENET_CLASSES = None

    def __init__(self, model_name: str = "vit_b_16", device: str = "auto", huggingface_model: str = ""):
        """
        Inicializa el servicio de visión

        Args:
            model_name: Nombre del modelo torchvision ('vit_b_16', 'efficientnet_b0', 'resnet50')
            device: 'cuda', 'cpu' o 'auto' (detecta automáticamente)
            huggingface_model: Modelo de HuggingFace Hub (ej: "google/vit-base-patch16-224")
        """
        self.model_name = model_name
        self.huggingface_model = huggingface_model
        self.use_huggingface = bool(huggingface_model)
        self.processor = None  # Para modelos de HuggingFace

        # Configurar dispositivo
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Inicializando modelo {huggingface_model if self.use_huggingface else model_name} en dispositivo: {self.device}")

        # Cargar modelo preentrenado
        if self.use_huggingface:
            self.model, self.processor = self._load_huggingface_model(huggingface_model)
        else:
            self.model = self._load_model(model_name)

        self.model.eval()
        self.model.to(self.device)

        # Transformaciones estándar para ImageNet (solo si no es HuggingFace)
        if not self.use_huggingface:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        # Cargar nombres de clases de ImageNet
        self._load_imagenet_classes()

        logger.info(f"✅ Modelo {huggingface_model if self.use_huggingface else model_name} cargado exitosamente")

    def _load_model(self, model_name: str) -> torch.nn.Module:
        """Carga el modelo especificado desde torchvision"""
        if model_name == "vit_b_16":
            return models.vit_b_16(pretrained=True)
        elif model_name == "efficientnet_b0":
            return models.efficientnet_b0(pretrained=True)
        elif model_name == "resnet50":
            return models.resnet50(pretrained=True)
        else:
            logger.warning(f"Modelo {model_name} no reconocido, usando ViT-B/16")
            return models.vit_b_16(pretrained=True)

    def _load_huggingface_model(self, model_id: str):
        """Carga el modelo desde HuggingFace Hub"""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            logger.info(f"Cargando modelo desde HuggingFace Hub: {model_id}")

            # Cargar processor y modelo
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForImageClassification.from_pretrained(model_id)

            logger.info(f"✓ Modelo HuggingFace {model_id} cargado correctamente")
            return model, processor

        except ImportError:
            logger.error("transformers no está instalado. Instala con: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Error cargando modelo de HuggingFace: {e}")
            logger.info("Cayendo al modelo torchvision por defecto")
            # Fallback a torchvision
            self.use_huggingface = False
            return models.vit_b_16(pretrained=True), None

    def _load_imagenet_classes(self):
        """Carga los nombres de las clases de ImageNet"""
        try:
            # Intenta cargar desde archivo local
            import os
            classes_path = os.path.join(os.path.dirname(__file__), "imagenet_classes.json")

            if os.path.exists(classes_path):
                with open(classes_path, 'r') as f:
                    self.IMAGENET_CLASSES = json.load(f)
            else:
                # Fallback: usar números si no hay archivo
                logger.warning("Archivo imagenet_classes.json no encontrado, usando IDs numéricos")
                self.IMAGENET_CLASSES = {str(i): f"class_{i}" for i in range(1000)}
        except Exception as e:
            logger.error(f"Error cargando clases de ImageNet: {e}")
            self.IMAGENET_CLASSES = {str(i): f"class_{i}" for i in range(1000)}

    async def predict(self, image_path: str, top_k: int = 5) -> Dict:
        """
        Realiza inferencia sobre una imagen

        Args:
            image_path: Ruta a la imagen
            top_k: Número de predicciones a retornar

        Returns:
            Dict con predicciones, scores y metadata
        """
        start_time = time.time()

        try:
            # Cargar imagen
            image = Image.open(image_path).convert('RGB')
            original_size = image.size

            if self.use_huggingface and self.processor:
                # Usar HuggingFace processor y modelo
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Inferencia
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.nn.functional.softmax(logits[0], dim=0)

                # Top-K predicciones
                top_prob, top_catid = torch.topk(probabilities, top_k)

                # Formatear resultados usando las etiquetas del modelo
                predictions = []
                for prob, catid in zip(top_prob, top_catid):
                    class_id = catid.item()
                    # Usar etiquetas del modelo si están disponibles
                    class_name = self.model.config.id2label.get(class_id, f"class_{class_id}")

                    predictions.append({
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "probability": float(prob.item()),
                        "confidence_percent": float(prob.item() * 100)
                    })

            else:
                # Usar torchvision modelo
                tensor = self.transform(image).unsqueeze(0).to(self.device)

                # Inferencia
                with torch.no_grad():
                    output = self.model(tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)

                # Top-K predicciones
                top_prob, top_catid = torch.topk(probabilities, top_k)

                # Formatear resultados
                predictions = []
                for prob, catid in zip(top_prob, top_catid):
                    class_id = str(catid.item())
                    class_name = self.IMAGENET_CLASSES.get(class_id, f"class_{class_id}")

                    predictions.append({
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "probability": float(prob.item()),
                        "confidence_percent": float(prob.item() * 100)
                    })

            inference_time = time.time() - start_time

            model_used = self.huggingface_model if self.use_huggingface else self.model_name
            logger.info(f"Inferencia completada en {inference_time:.3f}s - Top prediction: {predictions[0]['class_name']} ({predictions[0]['confidence_percent']:.1f}%)")

            return {
                "predictions": predictions,
                "model": model_used,
                "device": str(self.device),
                "inference_time": inference_time,
                "image_size": original_size,
                "top_prediction": {
                    "class": predictions[0]["class_name"],
                    "confidence": predictions[0]["confidence_percent"]
                }
            }

        except Exception as e:
            logger.error(f"Error en inferencia: {e}")
            raise

    async def predict_batch(self, image_paths: List[str], top_k: int = 5) -> List[Dict]:
        """
        Realiza inferencia en múltiples imágenes (batching)

        Args:
            image_paths: Lista de rutas a imágenes
            top_k: Número de predicciones por imagen

        Returns:
            Lista de resultados de predicción
        """
        results = []

        for image_path in image_paths:
            result = await self.predict(image_path, top_k)
            results.append(result)

        return results

    def get_model_info(self) -> Dict:
        """Retorna información sobre el modelo cargado"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_type": type(self.model).__name__,
            "input_size": (224, 224),
            "num_classes": 1000
        }
