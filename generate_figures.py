"""Script pour générer les figures du rapport QSVM audio folklorique.

Les figures sont sauvegardées dans docs/rapport/figures/ et référencées dans memoire.tex.

Ce script suppose que :
- les données audio et les features ont déjà été générées (mode train exécuté),
- le fichier HDF5 de features existe : results/features/extracted_features.h5,
- les modèles SVM / QSVM sont entraînés (facultatif pour certaines figures).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from src.data_pipeline.core import AudioPreprocessor, PreprocessParams
from src.evaluation.core import ModelEvaluator
from src.pipeline.core import AudioQSVMpipeline


ROOT = Path(__file__).resolve().parents[0]
FIG_DIR = ROOT / "docs" / "rapport" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1 : pipeline global (à dessiner manuellement)
# ---------------------------------------------------------------------------
# Cette figure est conceptuelle (boîtes et flèches). Vous pouvez la dessiner
# dans un outil comme draw.io, Inkscape ou PowerPoint et l'exporter en PDF
# sous le nom : docs/rapport/figures/pipeline_global.pdf
# ---------------------------------------------------------------------------


def _choose_example_audio(raw_audio_dir: Path) -> Path:
    """Choisit un fichier audio d'exemple pour les figures de prétraitement.

    On prend simplement le premier fichier trouvé dans data/raw_audio/.
    """

    for label_dir in raw_audio_dir.iterdir():
        if not label_dir.is_dir():
            continue
        for f in sorted(label_dir.glob("*.wav")):
            return f
        for f in sorted(label_dir.glob("*.mp3")):
            return f
    raise FileNotFoundError(
        f"Aucun fichier audio trouvé dans {raw_audio_dir}. Placez des fichiers dans data/raw_audio/gurna/ ou non_gurna/."
    )


# ---------------------------------------------------------------------------
# Figure 2 : prétraitement (onde brute + segments)
# ---------------------------------------------------------------------------


def fig_pretraitement() -> None:
    """Génère une figure illustrant le prétraitement audio.

    - Signal brut
    - Signal concaténé sans silences
    - Positions approximatives des segments
    """

    pipeline = AudioQSVMpipeline(config_path="config/paths.yaml")
    raw_dir = pipeline.raw_audio_dir

    example_path = _choose_example_audio(raw_dir)

    audio_cfg = pipeline.config.audio
    params = PreprocessParams(
        sample_rate=int(audio_cfg.get("sample_rate", 22050)),
        segment_duration_seconds=float(audio_cfg.get("segment_duration_seconds", 8)),
        silence_top_db=float(audio_cfg.get("silence_top_db", 30)),
    )

    pre = AudioPreprocessor(raw_dir=raw_dir, processed_dir=pipeline.processed_audio_dir, params=params)

    # Signal brut
    y, sr = librosa.load(example_path, sr=params.sample_rate, mono=True)

    # Segments après suppression du silence
    y_norm = pre._normalize_rms(y, params.target_rms)  # type: ignore[attr-defined]
    segments = pre._split_and_trim_silence(y_norm)  # type: ignore[attr-defined]
    if segments:
        y_concat = np.concatenate(segments)
    else:
        y_concat = y_norm

    t_raw = np.linspace(0, len(y) / sr, num=len(y))
    t_proc = np.linspace(0, len(y_concat) / sr, num=len(y_concat))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    axes[0].plot(t_raw, y, color="steelblue")
    axes[0].set_title("Signal brut")
    axes[0].set_xlabel("Temps (s)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t_proc, y_concat, color="seagreen")
    axes[1].set_title("Signal après suppression des silences et concaténation")
    axes[1].set_xlabel("Temps (s)")
    axes[1].set_ylabel("Amplitude")

    fig.tight_layout()
    out = FIG_DIR / "pretraitement.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure prétraitement sauvegardée dans {out}")


# ---------------------------------------------------------------------------
# Figure 3 : PCA des features (projection 2D X_reduced)
# ---------------------------------------------------------------------------


def fig_pca_features() -> None:
    """Génère une projection PCA 2D des features pour visualiser les classes."""

    pipeline = AudioQSVMpipeline(config_path="config/paths.yaml")
    if not pipeline.features_h5_path.exists():
        raise FileNotFoundError(
            f"Fichier de features introuvable : {pipeline.features_h5_path}. Lancez d'abord main.py --mode train."
        )

    with h5py.File(pipeline.features_h5_path, "r") as f:
        X = np.array(f["X"], dtype=np.float32)
        y = np.array(f["y"], dtype=np.int64)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="coolwarm", alpha=0.7)
    ax.set_xlabel("Composante principale 1")
    ax.set_ylabel("Composante principale 2")
    ax.set_title("Projection PCA des features (2D)")
    handles, labels = scatter.legend_elements()
    ax.legend(handles, ["gurna", "non_gurna"], title="Classe")

    fig.tight_layout()
    out = FIG_DIR / "pca_features.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure PCA sauvegardée dans {out}")


# ---------------------------------------------------------------------------
# Figures de performance (matrices de confusion, ROC)
# ---------------------------------------------------------------------------


def fig_matrices_confusion_et_roc() -> None:
    """Utilise ModelEvaluator pour générer matrices de confusion et ROC.

    Construit un split train/test à partir de X, y puis évalue le SVM RBF
    et, si disponible, le QSVM.
    """

    pipeline = AudioQSVMpipeline(config_path="config/paths.yaml")
    if not pipeline.features_h5_path.exists():
        raise FileNotFoundError(
            f"Fichier de features introuvable : {pipeline.features_h5_path}. Lancez d'abord main.py --mode train."
        )

    with h5py.File(pipeline.features_h5_path, "r") as f:
        X = np.array(f["X"], dtype=np.float32)
        y = np.array(f["y"], dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SVM RBF
    svm = pipeline._load_svm_or_raise()
    y_pred_svm = svm.predict(X_test)
    y_proba_svm = svm.predict_proba(X_test)

    eval_dir = ROOT / "results" / "evaluations"
    evaluator = ModelEvaluator(results_dir=eval_dir)

    res_svm = evaluator.evaluate_model(
        y_true=y_test,
        y_pred=y_pred_svm,
        y_proba=y_proba_svm,
        model_name="svm_rbf",
    )
    print("SVM RBF metrics:", res_svm.metrics)

    # Copie des figures dans docs/rapport/figures
    cm_svm_src = res_svm.confusion_matrix_path
    roc_svm_src = res_svm.roc_curve_path
    if cm_svm_src is not None:
        (FIG_DIR / "cm_svm.pdf").write_bytes(cm_svm_src.read_bytes())
    if roc_svm_src is not None:
        (FIG_DIR / "roc_svm_qsvm.pdf").write_bytes(roc_svm_src.read_bytes())

    # QSVM (optionnel)
    if pipeline.qsvm_model_path.exists():
        from src.models.quantum_svm import QuantumSVM

        qsvm = QuantumSVM.load(pipeline.qsvm_model_path)
        y_pred_q = qsvm.predict(X_test)
        # Optionnellement, proba si activées dans le modèle
        try:
            y_proba_q = qsvm.predict_proba(X_test)
        except Exception:
            y_proba_q = None

        res_q = evaluator.evaluate_model(
            y_true=y_test,
            y_pred=y_pred_q,
            y_proba=y_proba_q,
            model_name="qsvm",
        )
        print("QSVM metrics:", res_q.metrics)

        cm_q_src = res_q.confusion_matrix_path
        if cm_q_src is not None:
            (FIG_DIR / "cm_qsvm.pdf").write_bytes(cm_q_src.read_bytes())


def main() -> None:
    """Génère les principales figures du rapport.

    Certaines figures conceptuelles (pipeline global, schéma QSVM) restent à
    dessiner manuellement et à placer dans docs/rapport/figures/.
    """

    fig_pretraitement()
    fig_pca_features()
    fig_matrices_confusion_et_roc()


if __name__ == "__main__":
    main()
