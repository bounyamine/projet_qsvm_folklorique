"""Tests de base pour le module d'évaluation."""

from pathlib import Path

import numpy as np

from src.evaluation import ModelEvaluator


def test_model_evaluator_generates_files(tmp_path: Path) -> None:
    results_dir = tmp_path / "evaluations"
    evaluator = ModelEvaluator(results_dir=results_dir)

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array(
        [
            [0.8, 0.2],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )

    result = evaluator.evaluate_model(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        model_name="svm_rbf",
    )

    assert "accuracy" in result.metrics
    assert result.confusion_matrix_path is not None
    assert result.confusion_matrix_path.exists()
    assert result.roc_curve_path is not None
    assert result.roc_curve_path.exists()
