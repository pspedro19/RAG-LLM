"""
Metrics Module - BLEU, ROUGE, Semantic Similarity Evaluation
"""

import time
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

# Try to import optional metrics libraries
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    METRICS_AVAILABLE = True
except ImportError:
    logger.warning("NLTK/ROUGE not installed. Install with: pip install nltk rouge-score")
    METRICS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    SEMANTIC_AVAILABLE = True
except ImportError:
    logger.warning("SentenceTransformers not installed. Install with: pip install sentence-transformers")
    SEMANTIC_AVAILABLE = False


class MetricsCollector:
    """Colector centralizado de métricas del sistema"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.session_start = time.time()

        # Cargar modelo para similitud semántica si está disponible
        if SEMANTIC_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Semantic similarity model loaded")
            except Exception as e:
                logger.warning(f"Could not load semantic model: {e}")
                self.semantic_model = None
        else:
            self.semantic_model = None

    # ==========================================
    # Performance Metrics
    # ==========================================

    def record_response_time(self, endpoint: str, duration: float):
        """Registra tiempo de respuesta por endpoint"""
        self.metrics[f"{endpoint}_response_time"].append(duration)

    def record_tokens(self, endpoint: str, prompt: int, completion: int, total: int):
        """Registra consumo de tokens"""
        self.metrics[f"{endpoint}_tokens_prompt"].append(prompt)
        self.metrics[f"{endpoint}_tokens_completion"].append(completion)
        self.metrics[f"{endpoint}_tokens_total"].append(total)

    def record_agent_decision(self, agent: str, decision: str, success: bool):
        """Registra decisiones de agentes"""
        self.metrics["agent_decisions"].append({
            "agent": agent,
            "decision": decision,
            "success": success,
            "timestamp": time.time()
        })

    def record_rag_metrics(self,
                          recall: float,
                          precision_at_k: float,
                          retrieval_time: float,
                          num_chunks: int):
        """Registra métricas de RAG"""
        self.metrics["rag_recall"].append(recall)
        self.metrics["rag_precision_at_k"].append(precision_at_k)
        self.metrics["rag_retrieval_time"].append(retrieval_time)
        self.metrics["rag_chunks_retrieved"].append(num_chunks)

    # ==========================================
    # Text Quality Metrics (BLEU/ROUGE)
    # ==========================================

    def calculate_bleu(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calcula BLEU score entre texto de referencia y generado.

        Args:
            reference: Texto de referencia (gold standard)
            hypothesis: Texto generado por el modelo

        Returns:
            Dict con BLEU-1, BLEU-2, BLEU-3, BLEU-4
        """
        if not METRICS_AVAILABLE:
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

        try:
            # Tokenizar
            reference_tokens = [reference.lower().split()]
            hypothesis_tokens = hypothesis.lower().split()

            # Smoothing para evitar 0s
            smoothie = SmoothingFunction().method4

            # Calcular BLEU para diferentes n-gramas
            bleu_1 = sentence_bleu(reference_tokens, hypothesis_tokens,
                                  weights=(1, 0, 0, 0), smoothing_function=smoothie)
            bleu_2 = sentence_bleu(reference_tokens, hypothesis_tokens,
                                  weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
            bleu_3 = sentence_bleu(reference_tokens, hypothesis_tokens,
                                  weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
            bleu_4 = sentence_bleu(reference_tokens, hypothesis_tokens,
                                  weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

            results = {
                "bleu_1": round(bleu_1, 4),
                "bleu_2": round(bleu_2, 4),
                "bleu_3": round(bleu_3, 4),
                "bleu_4": round(bleu_4, 4)
            }

            # Registrar
            for key, value in results.items():
                self.metrics[key].append(value)

            return results

        except Exception as e:
            logger.error(f"Error calculating BLEU: {e}")
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}

    def calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calcula ROUGE scores (precision, recall, F1).

        Args:
            reference: Texto de referencia
            hypothesis: Texto generado

        Returns:
            Dict con ROUGE-1, ROUGE-2, ROUGE-L scores
        """
        if not METRICS_AVAILABLE:
            return {
                "rouge1_precision": 0.0, "rouge1_recall": 0.0, "rouge1_f1": 0.0,
                "rouge2_precision": 0.0, "rouge2_recall": 0.0, "rouge2_f1": 0.0,
                "rougeL_precision": 0.0, "rougeL_recall": 0.0, "rougeL_f1": 0.0
            }

        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, hypothesis)

            results = {
                "rouge1_precision": round(scores['rouge1'].precision, 4),
                "rouge1_recall": round(scores['rouge1'].recall, 4),
                "rouge1_f1": round(scores['rouge1'].fmeasure, 4),
                "rouge2_precision": round(scores['rouge2'].precision, 4),
                "rouge2_recall": round(scores['rouge2'].recall, 4),
                "rouge2_f1": round(scores['rouge2'].fmeasure, 4),
                "rougeL_precision": round(scores['rougeL'].precision, 4),
                "rougeL_recall": round(scores['rougeL'].recall, 4),
                "rougeL_f1": round(scores['rougeL'].fmeasure, 4)
            }

            # Registrar
            for key, value in results.items():
                self.metrics[key].append(value)

            return results

        except Exception as e:
            logger.error(f"Error calculating ROUGE: {e}")
            return {
                "rouge1_precision": 0.0, "rouge1_recall": 0.0, "rouge1_f1": 0.0,
                "rouge2_precision": 0.0, "rouge2_recall": 0.0, "rouge2_f1": 0.0,
                "rougeL_precision": 0.0, "rougeL_recall": 0.0, "rougeL_f1": 0.0
            }

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similitud semántica usando embeddings.

        Args:
            text1: Primer texto
            text2: Segundo texto

        Returns:
            Similitud coseno (0-1)
        """
        if not self.semantic_model:
            return 0.0

        try:
            embeddings = self.semantic_model.encode([text1, text2], convert_to_tensor=True)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

            self.metrics["semantic_similarity"].append(similarity)
            return round(similarity, 4)

        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0

    # ==========================================
    # Statistics & Reporting
    # ==========================================

    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumen estadístico de todas las métricas"""
        summary = {
            "session_duration": time.time() - self.session_start,
            "metrics": {}
        }

        for metric_name, values in self.metrics.items():
            if not values:
                continue

            # Para métricas numéricas
            if isinstance(values[0], (int, float)):
                summary["metrics"][metric_name] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "total": np.sum(values) if "tokens" in metric_name or "time" in metric_name else None
                }
            else:
                # Para métricas categóricas
                summary["metrics"][metric_name] = {
                    "count": len(values),
                    "samples": values[:5]  # Primeras 5 muestras
                }

        return summary

    def reset(self):
        """Resetea todas las métricas"""
        self.metrics.clear()
        self.session_start = time.time()
        logger.info("Metrics reset")


# Singleton global
metrics_collector = MetricsCollector()
