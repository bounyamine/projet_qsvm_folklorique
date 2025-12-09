"""Tests de base pour le module quantum_module.

Ces tests sont volontairement simples et se contentent de vérifier la
construction de circuits et les dimensions des matrices de Gram.
Ils sont conditionnés à la présence de Qiskit.
"""

import numpy as np
import pytest

from src.quantum_module import angle_encoding_circuit, compute_quantum_gram_matrix
from src.quantum_module.core import QuantumKernelConfig


def test_angle_encoding_circuit_qubits() -> None:
    pytest.importorskip("qiskit")
    x = np.random.randn(5)
    n_qubits = 4
    qc = angle_encoding_circuit(x, n_qubits)
    assert qc.num_qubits == n_qubits


def test_quantum_gram_matrix_shapes() -> None:
    pytest.importorskip("qiskit")
    X_train = np.random.randn(3, 4)
    X_test = np.random.randn(2, 4)

    cfg = QuantumKernelConfig(n_qubits=4, shots=256)
    K_train, K_test = compute_quantum_gram_matrix(X_train, X_test, cfg)

    assert K_train.shape == (3, 3)
    assert K_test is not None
    assert K_test.shape == (2, 3)
