"""Outils d'évaluation pour les modèles SVM RBF et QSVM.

Ce module fournit la classe :

- ModelEvaluator :
  * calcule les métriques classiques de classification binaire
    (accuracy, precision, recall, f1, roc_auc),
  * génère et sauvegarde les matrices de confusion,
  * génère et sauvegarde les courbes ROC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationResult:
    """Résultats d'évaluation pour un modèle donné."""

    model_name: str
    metrics: Dict[str, float]
    confusion_matrix_path: Optional[Path]
    roc_curve_path: Optional[Path]


class ModelEvaluator:
    """Évalue des modèles de classification binaire et produit des visualisations.

    Attributes:
        results_dir: Répertoire racine des résultats (e.g. `results/evaluations`).
    """

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = results_dir
        self.cm_dir = self.results_dir / "confusion_matrices"
        self.roc_dir = self.results_dir / "roc_curves"
        self.cm_dir.mkdir(parents=True, exist_ok=True)
        self.roc_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Métriques
    # ------------------------------------------------------------------
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calcule les métriques de classification binaire.

        Args:
            y_true: Labels réels (0/1).
            y_pred: Prédictions (0/1).
            y_proba: Probabilités pour la classe positive (shape: [n_samples, 2]
                ou [n_samples]). Si fourni, permet de calculer roc_auc.

        Returns:
            Dictionnaire de métriques.
        """

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        metrics: Dict[str, float] = {}
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

        if y_proba is not None:
            proba = np.asarray(y_proba)
            if proba.ndim == 2 and proba.shape[1] == 2:
                # On suppose que la deuxième colonne correspond à la classe positive
                proba_pos = proba[:, 1]
            else:
                proba_pos = proba
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
            except ValueError:
                # Cas où une seule classe est présente dans y_true
                metrics["roc_auc"] = float("nan")

        return metrics

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
    ) -> Path:
        """Génère et sauvegarde la matrice de confusion.

        Returns:
            Chemin du fichier image généré.
        """

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        cm = confusion_matrix(y_true, y_pred)
        labels = ["classe_0", "classe_1"]

        fig, ax = plt.subplots(figsize=(4, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"Matrice de confusion - {model_name}")
        fig.tight_layout()

        out_path = self.cm_dir / f"cm_{model_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
    ) -> Path:
        """Génère et sauvegarde la courbe ROC.

        Returns:
            Chemin du fichier image généré.
        """

        y_true = np.asarray(y_true)
        proba = np.asarray(y_proba)
        if proba.ndim == 2 and proba.shape[1] == 2:
            proba_pos = proba[:, 1]
        else:
            proba_pos = proba

        try:
            fpr, tpr, _ = roc_curve(y_true, proba_pos)
            auc = roc_auc_score(y_true, proba_pos)
        except ValueError:
            # Si une seule classe présente, on trace une diagonale vide
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            auc = float("nan")

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="aléatoire")
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.set_title(f"Courbe ROC - {model_name}")
        ax.legend(loc="lower right")
        fig.tight_layout()

        out_path = self.roc_dir / f"roc_{model_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # ------------------------------------------------------------------
    # Interface haut niveau
    # ------------------------------------------------------------------
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        model_name: str,
    ) -> EvaluationResult:
        """Calcule métriques, matrice de confusion et ROC (si possible).

        Args:
            y_true: Labels réels.
            y_pred: Prédictions.
            y_proba: Probabilités associées (optionnel mais recommandé).
            model_name: Nom du modèle (utilisé dans les noms de fichiers).

        Returns:
            EvaluationResult avec chemins vers les figures générées.
        """

        metrics = self.compute_metrics(y_true, y_pred, y_proba)

        cm_path = self.plot_confusion_matrix(y_true, y_pred, model_name=model_name)

        roc_path: Optional[Path] = None
        if y_proba is not None:
            roc_path = self.plot_roc_curve(y_true, y_proba, model_name=model_name)

        return EvaluationResult(
            model_name=model_name,
            metrics=metrics,
            confusion_matrix_path=cm_path,
            roc_curve_path=roc_path,
        )
