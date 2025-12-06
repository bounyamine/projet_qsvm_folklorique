"""Implémentation d'un SVM avec noyau quantique (QuantumSVM).

Ce wrapper est compatible scikit-learn:

    model = QuantumSVM()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

En interne, il utilise un `sklearn.svm.SVC(kernel="precomputed")` et la
matrice de Gram quantique calculée via `quantum_module`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.svm import SVC

from src.quantum_module.core import QuantumKernelConfig, compute_quantum_gram_matrix


@dataclass
class QuantumSVMConfig:
    """Configuration haut niveau pour QuantumSVM."""

    n_qubits: int = 8
    shots: int = 1024
    backend_name: str = "aer_simulator"
    C: float = 1.0


class QuantumSVM:
    """SVM avec noyau quantique personnalisé (compatible scikit-learn).

    Attributs principaux :
    - `config` : paramètres quantiques et du SVM.
    - `svc_` : instance interne de `SVC(kernel="precomputed")` après `fit`.
    - `X_train_` : copie des features d'entraînement (après PCA/sScaling).
    """

    def __init__(self, config: Optional[QuantumSVMConfig] = None) -> None:
        self.config = config or QuantumSVMConfig()
        self.svc_: Optional[SVC] = None
        self.X_train_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # API de type scikit-learn
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumSVM":
        """Entraîne le QSVM sur des features déjà préparées.

        Args:
            X: Matrice de features (n_samples, d).
            y: Labels (n_samples,).

        Returns:
            self.
        """

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        qcfg = QuantumKernelConfig(
            n_qubits=self.config.n_qubits,
            shots=self.config.shots,
            backend_name=self.config.backend_name,
        )
        K_train, _ = compute_quantum_gram_matrix(X, None, qcfg)

        svc = SVC(kernel="precomputed", C=self.config.C, probability=True, random_state=42)
        svc.fit(K_train, y)

        self.svc_ = svc
        self.X_train_ = X
        return self

    def _check_is_fitted(self) -> None:
        if self.svc_ is None or self.X_train_ is None:
            raise RuntimeError("QuantumSVM non entraîné. Appelez fit(X, y) d'abord.")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Calcule la fonction de décision SVM sur des nouveaux points."""

        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float32)

        qcfg = QuantumKernelConfig(
            n_qubits=self.config.n_qubits,
            shots=self.config.shots,
            backend_name=self.config.backend_name,
        )
        _, K_test = compute_quantum_gram_matrix(self.X_train_, X, qcfg)
        assert K_test is not None

        return self.svc_.decision_function(K_test)  # type: ignore[union-attr]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédit les labels pour de nouveaux points."""

        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float32)

        qcfg = QuantumKernelConfig(
            n_qubits=self.config.n_qubits,
            shots=self.config.shots,
            backend_name=self.config.backend_name,
        )
        _, K_test = compute_quantum_gram_matrix(self.X_train_, X, qcfg)
        assert K_test is not None

        return self.svc_.predict(K_test)  # type: ignore[union-attr]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Renvoie les probabilités par classe si activées dans SVC."""

        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float32)

        qcfg = QuantumKernelConfig(
            n_qubits=self.config.n_qubits,
            shots=self.config.shots,
            backend_name=self.config.backend_name,
        )
        _, K_test = compute_quantum_gram_matrix(self.X_train_, X, qcfg)
        assert K_test is not None

        return self.svc_.predict_proba(K_test)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Sauvegarde le modèle QSVM sur disque."""

        self._check_is_fitted()
        payload: dict[str, Any] = {
            "config": self.config,
            "svc_": self.svc_,
            "X_train_": self.X_train_,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "QuantumSVM":
        """Charge un modèle QSVM sauvegardé."""

        payload = joblib.load(path)
        model = cls(config=payload["config"])
        model.svc_ = payload["svc_"]
        model.X_train_ = payload["X_train_"]
        return model
