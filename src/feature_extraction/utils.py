"""Fonctions utilitaires pour l'extraction de features audio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureScalingModels:
    """Contient les objets de scaling et de PCA."""

    scaler: StandardScaler
    pca: PCA


def save_features_h5(
    path: Path,
    X: np.ndarray,
    y: np.ndarray,
    file_ids: List[str],
    scaling_models: FeatureScalingModels,
) -> None:
    """Sauvegarde les features et métadonnées dans un fichier HDF5.

    Args:
        path: Chemin du fichier HDF5.
        X: Matrice de features (n_samples, n_features_reduced).
        y: Labels (n_samples,).
        file_ids: Identifiants (e.g. chemins relatifs des segments).
        scaling_models: Objets StandardScaler et PCA.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("y", data=y)
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("file_ids", data=np.array(file_ids, dtype=dt))

        grp = f.create_group("scaling")
        grp.create_dataset("scaler_mean", data=scaling_models.scaler.mean_)
        grp.create_dataset("scaler_scale", data=scaling_models.scaler.scale_)
        grp.create_dataset("pca_components", data=scaling_models.pca.components_)
        grp.create_dataset("pca_mean", data=scaling_models.pca.mean_)


def fit_scaler_pca(
    X: np.ndarray, n_components: int
) -> Tuple[np.ndarray, FeatureScalingModels]:
    """Applique StandardScaler puis PCA.

    Args:
        X: Matrice de features initiale.
        n_components: Nombre de composantes principales.

    Returns:
        (X_reduced, models) avec X réduit et les objets scaler/PCA.
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)

    return X_reduced, FeatureScalingModels(scaler=scaler, pca=pca)
