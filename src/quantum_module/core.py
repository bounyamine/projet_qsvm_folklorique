"""Cœur du module quantique : encodage angulaire et noyau QSVM.

On suit l'idée générale :

    |φ(x)⟩ = ⊗_i R_y(2·arcsin(x_i)) |0⟩

et le noyau quantique

    K_q(x1, x2) = |⟨φ(x1)|φ(x2)⟩|²

est estimé comme la probabilité de mesurer l'état |0...0⟩ après avoir
appliqué U(x1) puis U†(x2) sur |0...0⟩.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .constants import DEFAULT_BACKEND_NAME, DEFAULT_N_QUBITS, DEFAULT_SHOTS


@dataclass
class QuantumKernelConfig:
    """Paramètres pour le calcul du noyau quantique."""

    n_qubits: int = DEFAULT_N_QUBITS
    shots: int = DEFAULT_SHOTS
    backend_name: str = DEFAULT_BACKEND_NAME


def _get_backend(backend_name: str):
    """Retourne un backend Qiskit en fonction de son nom.

    L'import de Qiskit est fait localement pour éviter les erreurs côté
    installation si le module n'est pas encore présent.
    """

    from qiskit_aer import Aer

    try:
        backend = Aer.get_backend(backend_name)
    except Exception as exc:  # pragma: no cover - dépend de l'install Qiskit
        raise RuntimeError(f"Impossible d'obtenir le backend Qiskit '{backend_name}': {exc}") from exc
    return backend


def _prepare_angles(features: np.ndarray, n_qubits: int) -> np.ndarray:
    """Prépare les angles d'encodage à partir d'un vecteur de features.

    * Tronque ou zero-pad les features à n_qubits.
    * Normalise dans [-1, 1] via une échelle simple puis applique 2·arcsin.
    """

    x = np.asarray(features, dtype=np.float32).ravel()

    if len(x) < n_qubits:
        x = np.pad(x, (0, n_qubits - len(x)), mode="constant")
    elif len(x) > n_qubits:
        x = x[:n_qubits]

    # Mise à l'échelle grossière : on suppose que les features sont
    # approximativement centrées; on les écrête dans [-1, 1].
    max_abs = np.max(np.abs(x)) + 1e-12
    x_scaled = x / max_abs
    x_clipped = np.clip(x_scaled, -1.0, 1.0)

    angles = 2.0 * np.arcsin(x_clipped)
    return angles.astype(np.float32)


def angle_encoding_circuit(features: np.ndarray, n_qubits: int):
    """Crée un circuit Qiskit avec encodage angulaire.

    Args:
        features: Vecteur de features d'entrée.
        n_qubits: Nombre de qubits.

    Returns:
        qiskit.QuantumCircuit avec les rotations RY appropriées.
    """

    from qiskit import QuantumCircuit

    angles = _prepare_angles(features, n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i, theta in enumerate(angles):
        qc.ry(theta, i)
    return qc


def _overlap_kernel_for_pair(
    x1: np.ndarray,
    x2: np.ndarray,
    cfg: QuantumKernelConfig,
) -> float:
    """Calcule K_q(x1, x2) pour une paire de vecteurs.

    On applique U(x1) puis U†(x2) ≈ RY(-θ_2) sur chaque qubit et on
    estime la probabilité de mesurer |0...0⟩.
    """

    from qiskit import QuantumCircuit
    # `execute` a été déplacé / rendu indisponible dans certaines versions
    # récentes de Qiskit. On essaie un import direct, sinon on utilise
    # l'API moderne `backend.run(circuit, shots=...)`.
    try:
        from qiskit import execute  # type: ignore
        _HAS_QISKIT_EXECUTE = True
    except Exception:
        execute = None  # type: ignore
        _HAS_QISKIT_EXECUTE = False

    n_qubits = cfg.n_qubits
    backend = _get_backend(cfg.backend_name)

    angles1 = _prepare_angles(x1, n_qubits)
    angles2 = _prepare_angles(x2, n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Préparation |φ(x1)⟩
    for i, theta in enumerate(angles1):
        qc.ry(theta, i)

    # Application de U†(x2) ≈ RY(-θ_2)
    for i, theta in enumerate(angles2):
        qc.ry(-theta, i)

    qc.measure(range(n_qubits), range(n_qubits))

    if _HAS_QISKIT_EXECUTE:
        job = execute(qc, backend=backend, shots=cfg.shots)
        result = job.result()
    else:
        # API moderne : `backend.run` retourne un Job
        job = backend.run(qc, shots=cfg.shots)
        # La plupart des providers exposent `result()` sur le job
        try:
            result = job.result()
        except Exception:
            # si `result()` n'est pas disponible, tenter d'accéder au
            # résultat via `job.result()` quand même (fallback minimal)
            result = job.result()

    # get_counts peut accepter le circuit ou non selon la version
    try:
        counts = result.get_counts(qc)
    except Exception:
        try:
            counts = result.get_counts()
        except Exception:
            counts = {}

    zero_state = "0" * n_qubits
    prob_zero = counts.get(zero_state, 0) / cfg.shots
    return float(prob_zero)


def quantum_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    config: Optional[QuantumKernelConfig] = None,
) -> float:
    """Calcule le noyau quantique K_q(x1, x2).

    Args:
        x1: Vecteur de features 1.
        x2: Vecteur de features 2.
        config: Configuration du noyau quantique.

    Returns:
        Valeur du noyau quantique dans [0, 1].
    """

    cfg = config or QuantumKernelConfig()
    return _overlap_kernel_for_pair(x1, x2, cfg)


@dataclass
class QuantumGramComputer:
    """Calcul de matrices de Gram quantiques avec mémoisation simple."""

    config: QuantumKernelConfig
    cache: Dict[Tuple[Tuple[float, ...], Tuple[float, ...]], float] = field(default_factory=dict)

    def _key(self, x1: np.ndarray, x2: np.ndarray) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        a = tuple(np.asarray(x1, dtype=float).ravel())
        b = tuple(np.asarray(x2, dtype=float).ravel())
        return (a, b) if a <= b else (b, a)

    def kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        key = self._key(x1, x2)
        if key in self.cache:
            return self.cache[key]
        val = _overlap_kernel_for_pair(x1, x2, self.config)
        self.cache[key] = val
        return val

    def compute_gram_matrix(
        self,
        X_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Calcule la matrice de Gram train-train (et train-test optionnelle).

        Args:
            X_train: Matrice (n_train, d).
            X_test: Optionnellement matrice (n_test, d).

        Returns:
            (K_train, K_test) où K_test peut être None.
        """

        X_train = np.asarray(X_train, dtype=np.float32)
        n_train = X_train.shape[0]

        K_train = np.zeros((n_train, n_train), dtype=np.float32)
        for i in range(n_train):
            K_train[i, i] = 1.0
            for j in range(i + 1, n_train):
                val = self.kernel(X_train[i], X_train[j])
                K_train[i, j] = K_train[j, i] = val

        K_test: Optional[np.ndarray] = None
        if X_test is not None:
            X_test = np.asarray(X_test, dtype=np.float32)
            n_test = X_test.shape[0]
            K_test = np.zeros((n_test, n_train), dtype=np.float32)
            for i in range(n_test):
                for j in range(n_train):
                    K_test[i, j] = self.kernel(X_test[i], X_train[j])

        return K_train, K_test


def compute_quantum_gram_matrix(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    config: Optional[QuantumKernelConfig] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Fonction utilitaire de haut niveau pour la matrice de Gram quantique."""

    cfg = config or QuantumKernelConfig()
    computer = QuantumGramComputer(cfg)
    return computer.compute_gram_matrix(X_train, X_test)
