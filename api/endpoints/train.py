"""Endpoint de réentraînement des modèles.

- POST /train
  * (Optionnel) protégé par une clé API fournie dans l'en-tête
    X-API-Key. Si la variable d'environnement QSVM_ADMIN_KEY est
    définie, la clé doit correspondre.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Header, HTTPException

from api.schemas import TrainRequest, TrainResponse
from src.pipeline import AudioQSVMpipeline


logger = logging.getLogger(__name__)

router = APIRouter()


def _check_api_key(x_api_key: str | None) -> None:
    """Vérifie la clé API d'administration si configurée.

    Si QSVM_ADMIN_KEY n'est pas défini, aucune protection n'est
    appliquée (convenable pour un usage local / expérimental).
    """

    expected = os.getenv("QSVM_ADMIN_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Clé API invalide ou manquante.")


@router.post("/train", response_model=TrainResponse)
async def train_endpoint(
    body: TrainRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> TrainResponse:
    """Réentraîne les modèles SVM RBF et QSVM.

    Args:
        body: Paramètres de la requête (actuellement `force_rebuild`).
        x_api_key: Clé API d'administration.
    """

    _check_api_key(x_api_key)

    pipeline = AudioQSVMpipeline(config_path="config/paths.yaml")

    # Pour l'instant, `force_rebuild` ne fait que logger une intention.
    if body.force_rebuild:
        logger.info("Réentraînement demandé avec force_rebuild=True")

    pipeline.train()
    metrics = pipeline.evaluate()

    return TrainResponse(message="Réentraînement terminé", metrics=metrics)
