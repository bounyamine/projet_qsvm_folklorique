"""Tests basiques pour les modèles, en particulier QuantumSVM.

Ces tests supposent que Qiskit est installé.
"""

import numpy as np
import pytest

from src.models import QuantumSVM
from src.models.quantum_svm import QuantumSVMConfig


def test_quantum_svm_fit_and_predict_small_dataset() -> None:
    pytest.importorskip("qiskit")
    # Petit dataset 2D linéairement séparable
    X = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.1],
            [0.9, 0.8],
            [0.8, 0.9],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1])

    cfg = QuantumSVMConfig(n_qubits=2, shots=256)
    model = QuantumSVM(config=cfg)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == y.shape
