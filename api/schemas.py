"""Schémas Pydantic pour l'API QSVM audio.

Ces modèles définissent la forme des réponses renvoyées par les
endpoints FastAPI (prédiction, informations modèles, réentraînement).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class ModelPrediction(BaseModel):
    """Prédiction d'un modèle (classique ou quantique)."""

    label: int
    probabilities: List[float]


class PredictionResponse(BaseModel):
    """Réponse de l'endpoint /predict."""

    svm_rbf: ModelPrediction
    qsvm: Optional[ModelPrediction] = None
    n_segments: int


class ModelInfoResponse(BaseModel):
    """Informations sur l'état des modèles et leurs performances."""

    svm_rbf_available: bool
    qsvm_available: bool
    metrics: Optional[Dict[str, float]] = None


class TrainRequest(BaseModel):
    """Corps de requête pour l'endpoint /train."""

    force_rebuild: bool = False


class TrainResponse(BaseModel):
    """Réponse de l'endpoint /train."""

    message: str
    metrics: Optional[Dict[str, float]] = None
