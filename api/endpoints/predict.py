"""Endpoint de prédiction pour l'API QSVM audio.

- POST /predict
  * Upload d'un fichier audio via multipart/form-data.
  * Utilise AudioQSVMpipeline.predict pour obtenir les prédictions
    SVM RBF et QSVM.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import ModelPrediction, PredictionResponse
from src.pipeline import AudioQSVMpipeline


logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)) -> PredictionResponse:
    """Prédit le genre (Gurna vs non-Gurna) pour un fichier audio.

    Le fichier est temporairement enregistré sur disque, passé au
    pipeline d'inférence, puis supprimé.
    """

    if not file.filename:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni.")

    suffix = Path(file.filename).suffix or ".wav"
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        await file.close()

    try:
        pipeline = AudioQSVMpipeline(config_path="config/paths.yaml")
        raw_result = pipeline.predict(str(tmp_path))
    except FileNotFoundError as exc:
        logger.exception("Erreur de fichier audio introuvable")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Erreur dans le pipeline de prédiction")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:  # pragma: no cover - nettoyage best-effort
            logger.warning("Impossible de supprimer le fichier temporaire %s", tmp_path)

    svm_raw = raw_result.get("svm_rbf")
    qsvm_raw = raw_result.get("qsvm")

    if not svm_raw:
        raise HTTPException(
            status_code=500, detail="Résultat SVM RBF manquant dans le pipeline."
        )

    svm_pred = ModelPrediction(**svm_raw)
    qsvm_pred = ModelPrediction(**qsvm_raw) if qsvm_raw is not None else None

    return PredictionResponse(
        svm_rbf=svm_pred,
        qsvm=qsvm_pred,
        n_segments=int(raw_result.get("n_segments", 0)),
    )
