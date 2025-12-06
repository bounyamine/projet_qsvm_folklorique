"""Tests de base pour le module feature_extraction."""

from pathlib import Path

import numpy as np
import soundfile as sf

from src.feature_extraction import FeatureExtractor
from src.feature_extraction.core import FeatureParams


def _create_dummy_wav(path: Path, sr: int = 22050, duration: float = 1.0) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)


def test_feature_extraction_and_build_dataset(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed_audio"
    gurna_dir = processed_root / "gurna"
    wav_path = gurna_dir / "sample_seg000.wav"

    _create_dummy_wav(wav_path)

    params = FeatureParams(sample_rate=22050, pca_components=4)
    extractor = FeatureExtractor(params=params)

    label_mapping = {"gurna": 0}
    h5_path = tmp_path / "features.h5"

    extractor.build_and_save_dataset(processed_root, label_mapping, h5_path)

    assert h5_path.exists()
