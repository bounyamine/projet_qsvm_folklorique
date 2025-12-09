"""Extraction de features audio (MFCC, chroma, spectral, tempo, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np

from .constants import (
    MFCC_DIM,
)
from .utils import fit_scaler_pca, save_features_h5


@dataclass
class FeatureParams:
    """Paramètres de l'extraction de features."""

    sample_rate: int = 22050
    n_mfcc: int = MFCC_DIM
    hop_length: int = 512
    n_fft: int = 2048
    pca_components: int = 8


class FeatureExtractor:
    """Extraction de caractéristiques audio basées sur librosa.

    Cette classe convertit un fichier audio prétraité en un vecteur de
    features résumées (moyennes / écarts types dans le temps) puis
    applique un scaling + PCA pour produire des features de dimension
    réduite adaptées au QSVM.
    """

    def __init__(self, params: FeatureParams | None = None) -> None:
        self.params = params or FeatureParams()

    # ------------------------------------------------------------------
    # Extraction bas niveau
    # ------------------------------------------------------------------
    def _extract_raw_features(self, path: Path) -> Dict[str, np.ndarray]:
        """Extrait les cartes de features temporelles pour un fichier.

        Args:
            path: Chemin du segment audio prétraité.

        Returns:
            Dictionnaire de features (nom -> matrice [dim, frames]).
        """

        y, sr = librosa.load(path=str(path), sr=self.params.sample_rate, mono=True)

        hop = self.params.hop_length
        n_fft = self.params.n_fft

        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.params.n_mfcc, hop_length=hop, n_fft=n_fft
        )
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop, n_fft=n_fft)
        spec_contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=hop, n_fft=n_fft
        )
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)
        rms = librosa.feature.rms(y=y, hop_length=hop)
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=hop, n_fft=n_fft
        )
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=hop, n_fft=n_fft
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=hop, n_fft=n_fft
        )
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        return {
            "mfcc": mfcc,
            "chroma": chroma,
            "spec_contrast": spec_contrast,
            "tonnetz": tonnetz,
            "zcr": zcr,
            "rms": rms,
            "centroid": centroid,
            "bandwidth": bandwidth,
            "rolloff": rolloff,
            "tempo": np.array([[tempo]], dtype=np.float32),
        }

    @staticmethod
    def _aggregate_features(feats: Dict[str, np.ndarray]) -> np.ndarray:
        """Agrège les features temporelles en un vecteur 1D.

        On concatène pour chaque feature les moyennes et écarts types sur
        l'axe temps.
        """

        parts: List[np.ndarray] = []
        for name, mat in feats.items():
            if name == "tempo":
                parts.append(mat.flatten())
                continue
            # mat shape: (dim, frames)
            mean = mat.mean(axis=1)
            std = mat.std(axis=1)
            parts.append(mean)
            parts.append(std)
        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def extract_features_for_file(
        self, path: Path, label: int
    ) -> Tuple[np.ndarray, int, str]:
        """Extrait un vecteur de features pour un segment.

        Args:
            path: Chemin du segment audio.
            label: Label numérique associé.

        Returns:
            (feature_vector, label, file_id)
        """

        raw = self._extract_raw_features(path)
        vec = self._aggregate_features(raw)
        return vec, label, str(path)

    def build_dataset_from_directory(
        self, processed_root: Path, label_mapping: Dict[str, int]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Construit un dataset de features à partir de processed_root.

        Args:
            processed_root: Dossier racine contenant des sous-dossiers par classe.
            label_mapping: Mapping label_str -> label_int.

        Returns:
            (X, y, file_ids)
        """

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        file_ids: List[str] = []

        for label_str, label_int in label_mapping.items():
            class_dir = processed_root / label_str
            if not class_dir.exists():
                continue
            for wav_path in sorted(class_dir.glob("*.wav")):
                vec, lab, fid = self.extract_features_for_file(wav_path, label_int)
                X_list.append(vec)
                y_list.append(lab)
                file_ids.append(fid)

        if not X_list:
            raise RuntimeError(
                f"Aucun segment audio trouvé sous {processed_root}; vérifiez le prétraitement."
            )

        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)
        return X, y, file_ids

    def build_and_save_dataset(
        self,
        processed_root: Path,
        label_mapping: Dict[str, int],
        output_h5: Path,
    ) -> None:
        """Construit le dataset de features, applique scaling+PCA et sauvegarde.

        Args:
            processed_root: Racine des segments audio.
            label_mapping: Mapping label_str -> label_int.
            output_h5: Fichier HDF5 de sortie.
        """

        X, y, file_ids = self.build_dataset_from_directory(
            processed_root, label_mapping
        )
        X_reduced, models = fit_scaler_pca(X, n_components=self.params.pca_components)
        save_features_h5(output_h5, X_reduced, y, file_ids, models)
