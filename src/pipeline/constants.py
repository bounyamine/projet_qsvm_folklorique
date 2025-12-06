"""Constantes pour le module de pipeline QSVM.

Contient notamment le mapping entre labels textuels et entiers.
"""

from __future__ import annotations

from typing import Dict

# Mapping par défaut pour la classification binaire folklorique
LABEL_MAPPING: Dict[str, int] = {
    "gurna": 0,
    "non_gurna": 1,
}