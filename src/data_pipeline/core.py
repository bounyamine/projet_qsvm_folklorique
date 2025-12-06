"""Prétraitement audio : conversion, segmentation, suppression du silence, normalisation.

Ce module ne dépend que de paramètres simples (taux d'échantillonnage,
seuil de silence, durée de segment) pour rester modulaire. L'intégration
avec les fichiers YAML de configuration se fera dans le pipeline haut
niveau.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import librosa
import numpy as np

from .constants import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEGMENT_DURATION_SECONDS,
    DEFAULT_SILENCE_TOP_DB,
    DEFAULT_TARGET_RMS,
)
from .utils import iter_audio_files, write_wav


@dataclass
class PreprocessParams:
    """Paramètres du prétraitement audio."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    segment_duration_seconds: float = DEFAULT_SEGMENT_DURATION_SECONDS
    silence_top_db: float = DEFAULT_SILENCE_TOP_DB
    target_rms: float = DEFAULT_TARGET_RMS


class AudioPreprocessor:
    """Pipeline de prétraitement audio.

    Responsabilités :
    - Charger des fichiers audio depuis `raw_dir`.
    - Convertir en WAV mono, sr fixe.
    - Supprimer les silences et découper en segments fixes.
    - Normaliser le volume (RMS).
    - Sauvegarder les segments dans `processed_dir/label/`.
    """

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        params: PreprocessParams | None = None,
    ) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.params = params or PreprocessParams()

    # ------------------------------------------------------------------
    # Opérations de base
    # ------------------------------------------------------------------
    def _load_mono(self, path: Path) -> np.ndarray:
        """Charge un fichier audio et renvoie un signal mono normalisé.

        Args:
            path: Chemin du fichier audio.

        Returns:
            Signal audio mono.
        """

        y, _ = librosa.load(path=str(path), sr=self.params.sample_rate, mono=True)
        return y.astype(np.float32)

    @staticmethod
    def _normalize_rms(y: np.ndarray, target_rms: float) -> np.ndarray:
        """Normalise l'énergie RMS du signal.

        Args:
            y: Signal audio.
            target_rms: RMS cible.

        Returns:
            Signal normalisé.
        """

        rms = np.sqrt(np.mean(y**2) + 1e-12)
        if rms < 1e-8:
            return y
        gain = target_rms / rms
        return (y * gain).astype(np.float32)

    def _split_and_trim_silence(self, y: np.ndarray) -> List[np.ndarray]:
        """Supprime les silences et découpe en segments de durée fixe.

        Args:
            y: Signal audio mono.

        Returns:
            Liste de segments audio.
        """

        sample_rate = self.params.sample_rate
        segment_len = int(self.params.segment_duration_seconds * sample_rate)

        # Découpage des zones non silencieuses
        intervals = librosa.effects.split(y, top_db=self.params.silence_top_db)
        non_silent = []
        for start, end in intervals:
            non_silent.append(y[start:end])

        if not non_silent:
            return []

        y_concat = np.concatenate(non_silent)

        segments: List[np.ndarray] = []
        for start in range(0, len(y_concat), segment_len):
            seg = y_concat[start : start + segment_len]
            if len(seg) < segment_len // 2:
                # On ignore les très petits restes
                continue
            segments.append(seg)

        return segments

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def process_file(self, src_path: Path, label: str) -> List[Path]:
        """Prétraite un fichier audio et renvoie les chemins des segments.

        Args:
            src_path: Chemin du fichier audio (dans `raw_dir` ou arbitraire).
            label: Nom de la classe (ex. "gurna", "non_gurna", ou label spécial).

        Returns:
            Liste des chemins vers les segments prétraités.
        """

        y = self._load_mono(src_path)
        y = self._normalize_rms(y, self.params.target_rms)
        segments = self._split_and_trim_silence(y)

        out_dir = self.processed_dir / label
        out_dir.mkdir(parents=True, exist_ok=True)

        out_paths: List[Path] = []
        stem = src_path.stem
        for i, seg in enumerate(segments):
            out_path = out_dir / f"{stem}_seg{i:03d}.wav"
            write_wav(out_path, seg, self.params.sample_rate)
            out_paths.append(out_path)

        return out_paths

    def run_full_pipeline(self) -> None:
        """Traite tous les fichiers sous `raw_dir`.

        On attend une structure du type :

        raw_dir/
          gurna/*.wav
          non_gurna/*.wav
        """

        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Répertoire raw_dir introuvable: {self.raw_dir}")

        for label_dir in self.raw_dir.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            files = iter_audio_files(label_dir)
            for f in files:
                self.process_file(f, label=label)
