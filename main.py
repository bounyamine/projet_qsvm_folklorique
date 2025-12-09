"""Pipeline principal du projet QSVM pour classification audio.

Usage:
    python main.py --config config/paths.yaml --mode train
    python main.py --config config/paths.yaml --mode predict --audio path/to/audio.wav
    python main.py --config config/paths.yaml --mode evaluate
"""

import argparse
import logging
import time
from pathlib import Path

from src.pipeline import AudioQSVMpipeline
from src.utils.progress_ui import (
    print_banner,
    print_error,
    print_info,
    print_result,
    print_section,
    print_step,
    print_success,
    print_warning,
    ProgressStage,
)
from src.utils.progress_logging import configure_live_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QSVM Audio Classification")
    parser.add_argument("--config", required=True, help="Path to paths config file (YAML)")
    parser.add_argument("--mode", choices=["train", "predict", "evaluate"], required=True)
    parser.add_argument("--audio", help="Audio file for prediction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configurer le logging en temps réel
    configure_live_logging(level=logging.INFO)

    print_section(
        "🎵 QSVM Audio Classification Pipeline",
        "Quantum SVM for Cameroon Folklorique Music",
    )

    start_time = time.time()

    try:
        pipeline = AudioQSVMpipeline(config_path=args.config)

        if args.mode == "train":
            _run_train_mode(pipeline)

        elif args.mode == "predict":
            if not args.audio:
                print_error("Le mode predict requiert --audio")
                raise SystemExit(
                    "Invalid argument: for predict mode, --audio is required."
                )
            _run_predict_mode(pipeline, args.audio)

        elif args.mode == "evaluate":
            _run_evaluate_mode(pipeline)

        else:
            print_error("Mode invalide")
            raise SystemExit("Invalid mode")

        elapsed = time.time() - start_time
        print_success(
            f"Pipeline complété avec succès en {elapsed:.2f} secondes!"
        )

    except Exception as exc:
        elapsed = time.time() - start_time
        print_error(f"{exc}")
        raise SystemExit(f"Pipeline échoué après {elapsed:.2f}s: {exc}") from exc


def _run_train_mode(pipeline: AudioQSVMpipeline) -> None:
    """Exécute le mode entraînement avec progression visuelle."""
    print_banner("Mode ENTRAÎNEMENT activé", ProgressStage.TRAINING)
    print_section("Entraînement en cours")

    logger = logging.getLogger("src.pipeline")
    logger.info("[Train] Démarrage du mode entraînement...")

    pipeline.train()

    print_result("Modèles sauvegardés", f"→ {pipeline.models_dir}")
    print_success("Entraînement terminé!")


def _run_predict_mode(pipeline: AudioQSVMpipeline, audio_path: str) -> None:
    """Exécute le mode prédiction avec progression visuelle."""
    audio_file = Path(audio_path)

    print_banner("Mode PRÉDICTION activé", ProgressStage.PREDICTION)
    print_section("Prédiction sur fichier audio")

    if not audio_file.exists():
        print_error(f"Fichier audio introuvable: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print_info(f"Fichier audio: {audio_file.name}")

    # Exécuter la prédiction
    print_step(1, 1, "Traitement audio et prédiction en cours...")
    logger = logging.getLogger("src.pipeline")
    logger.info(f"[Predict] Prédiction sur {audio_file.name}...")
    result = pipeline.predict(audio_path)

    print_section("Résultats de Prédiction")
    svm_result = result.get("svm_rbf", {})
    svm_label = svm_result.get("label")
    svm_prob = svm_result.get("probabilities", [0, 0])
    label_text = "🎵 Gurna" if svm_label == 1 else "🎵 Non-Gurna"

    print_result("Modèle SVM RBF", label_text, "\033[92m")
    print_result("  Confiance", f"{max(svm_prob) * 100:.1f}%")
    print_result("  Segments analysés", result.get("n_segments", "N/A"))

    if result.get("qsvm"):
        q_result = result["qsvm"]
        q_label = q_result.get("label")
        q_prob = q_result.get("probabilities", [0, 0])
        q_label_text = "🎵 Gurna" if q_label == 1 else "🎵 Non-Gurna"
        print_result("Modèle QSVM", q_label_text, "\033[94m")
        print_result("  Confiance", f"{max(q_prob) * 100:.1f}%")

    print_success("Prédiction terminée!")


def _run_evaluate_mode(pipeline: AudioQSVMpipeline) -> None:
    """Exécute le mode évaluation avec progression visuelle."""
    print_banner("Mode ÉVALUATION activé", ProgressStage.EVALUATION)
    print_section("Évaluation des modèles")

    logger = logging.getLogger("src.pipeline")
    logger.info("[Eval] Démarrage de l'évaluation...")
    metrics = pipeline.evaluate()

    print_section("Métriques d'Évaluation")
    if metrics:
        for metric_name, metric_value in metrics.items():
            print_result(
                metric_name.replace("_", " ").title(),
                f"{metric_value:.4f}",
            )
    print_success("Évaluation terminée!")


if __name__ == "__main__":
    main()
