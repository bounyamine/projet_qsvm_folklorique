"""Fonctions utilitaires pour le prétraitement audio."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import soundfile as sf


def write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    """Écrit un signal audio mono en WAV.

    Args:
        path: Chemin de sortie.
        y: Signal audio mono.
        sr: Taux d'échantillonnage.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(file=str(path), data=y, samplerate=sr)


def iter_audio_files(root: Path, extensions: Iterable[str] = (".wav", ".mp3", ".flac")) -> List[Path]:
    """Retourne récursivement tous les fichiers audio sous un répertoire.

    Args:
        root: Répertoire racine.
        extensions: Extensions supportées.

    Returns:
        Liste de chemins de fichiers audio.
    """

    exts = {e.lower() for e in extensions}
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]
