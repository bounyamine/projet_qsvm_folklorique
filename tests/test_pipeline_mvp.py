"""Tests de base pour le MVP AudioQSVMpipeline."""

from pathlib import Path

from src.pipeline import AudioQSVMpipeline


def test_pipeline_train_and_evaluate(tmp_path: Path) -> None:
    """Vérifie que le pipeline MVP s'entraîne et s'évalue sans erreur."""

    # On copie paths.yaml dans un dossier temporaire pour le test si besoin,
    # mais ici on suppose que le fichier existe à l'emplacement par défaut.
    config_path = Path("config/paths.yaml")
    assert config_path.exists()

    pipeline = AudioQSVMpipeline(config_path=str(config_path))
    pipeline.train()
    metrics = pipeline.evaluate()

    assert "svm_rbf_accuracy" in metrics
    assert 0.0 <= metrics["svm_rbf_accuracy"] <= 1.0
