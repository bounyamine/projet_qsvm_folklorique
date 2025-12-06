"""Module quantique : encodage angulaire et noyau quantique QSVM."""

from .core import angle_encoding_circuit, compute_quantum_gram_matrix, quantum_kernel

__all__ = [
    "angle_encoding_circuit",
    "quantum_kernel",
    "compute_quantum_gram_matrix",
]
