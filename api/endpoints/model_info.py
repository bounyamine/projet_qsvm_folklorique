"""Endpoint d'information sur les modèles QSVM/SVM.

- GET /model_info
  * Indique si les modèles SVM RBF et QSVM sont disponibles.
  * Renvoie les métriques d'évaluation en appelant le pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter

from api.schemas import ModelInfoResponse
from src.pipeline import AudioQSVMpipeline


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/model_info", response_model=ModelInfoResponse)
async def model_info_endpoint() -> ModelInfoResponse:
    """Renvoie le statut des modèles et, si possible, leurs métriques.

    Pour l'instant, on appelle directement `pipeline.evaluate()` ce qui
    peut être coûteux sur de grandes bases. Pour un usage intensif, il
    faudra persister les métriques à l'entraînement.
    """

    pipeline = AudioQSVMpipeline(config_path="config/paths.yaml")

    svm_exists = Path(pipeline.svm_model_path).exists()
    qsvm_exists = Path(pipeline.qsvm_model_path).exists()

    metrics = None
    if svm_exists:
        try:
            metrics = pipeline.evaluate()
        except Exception as exc:  # pragma: no cover
            logger.exception("Erreur lors de l'évaluation des modèles")

    return ModelInfoResponse(
        svm_rbf_available=svm_exists,
        qsvm_available=qsvm_exists,
        metrics=metrics,
    )
