"""Application FastAPI pour le projet QSVM audio folklorique.

Endpoints principaux:
- POST /predict : prédiction sur un fichier audio uploadé.
- GET  /model_info : statut des modèles et métriques de performance.
- POST /train : réentraînement des modèles (protégé par clé API optionnelle).
"""

from __future__ import annotations

import logging
from fastapi import FastAPI

from api.endpoints import model_info, predict, train


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="QSVM Audio Classification API",
    description=(
        "API pour la classification binaire de musique folklorique camerounaise "
        "(Gurna vs non-Gurna) avec SVM classique et QSVM quantique."
    ),
    version="0.1.0",
)


app.include_router(predict.router, tags=["prediction"])
app.include_router(model_info.router, tags=["model_info"])
app.include_router(train.router, tags=["train"])


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Endpoint de santé simple pour vérifier que l'API tourne."""

    return {"status": "ok"}
