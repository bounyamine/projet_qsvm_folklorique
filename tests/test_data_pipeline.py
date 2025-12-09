"""Tests de base pour le module data_pipeline."""

from pathlib import Path

import numpy as np
import soundfile as sf

from src.data_pipeline import AudioPreprocessor
from src.data_pipeline.core import PreprocessParams


def _create_dummy_wav(path: Path, sr: int = 22050, duration: float = 2.0) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)


def test_preprocessor_process_file(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_audio/gurna"
    processed_dir = tmp_path / "processed_audio"
    src = raw_dir / "gurna002.wav"

    _create_dummy_wav(src)

    params = PreprocessParams(segment_duration_seconds=0.5)
    pre = AudioPreprocessor(
        raw_dir=tmp_path / "raw_audio", processed_dir=processed_dir, params=params
    )

    out_paths = pre.process_file(src, label="gurna")

    assert out_paths
    for p in out_paths:
        assert p.exists()
