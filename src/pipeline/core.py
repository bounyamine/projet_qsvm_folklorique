"""Implémentation du pipeline AudioQSVMpipeline.

Ce module orchestre l'ensemble du pipeline :
- Prétraitement audio (data_pipeline) si des données sont disponibles.
- Extraction de features + PCA (feature_extraction).
- Entraînement et comparaison SVM RBF classique vs QSVM (QuantumSVM).
- Fallback possible sur un dataset synthétique si aucune donnée audio
  n'est présente, pour garder un pipeline exécutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import joblib
import logging
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.data_pipeline.core import AudioPreprocessor, PreprocessParams
from src.models import QuantumSVM
from src.models.quantum_svm import QuantumSVMConfig
from src.pipeline.constants import LABEL_MAPPING


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration globale du pipeline chargée depuis les fichiers YAML.

    Attributs :
    - root_dir : racine du projet
    - paths   : contenu de config/paths.yaml
    - audio   : contenu de config/audio_params.yaml
    - quantum : contenu de config/quantum_params.yaml
    """

    root_dir: Path
    paths: Dict[str, Any]
    audio: Dict[str, Any]
    quantum: Dict[str, Any]


class AudioQSVMpipeline:
    """Pipeline haut niveau pour l'entraînement, la prédiction et l'évaluation.

    Priorité :
    - Utiliser de vraies données audio si elles sont présentes dans
      `data/raw_audio/<label>/`.
    - Sinon, retomber sur un dataset synthétique pour garder un pipeline
      fonctionnel.
    - Entraîner et comparer SVM RBF (classique) et QSVM.
    """

    def __init__(self, config_path: str) -> None:
        self.config = self._load_config(Path(config_path))

        # Résolution des chemins importants à partir de paths.yaml
        paths_cfg = self.config.paths["data"]
        root = self.config.root_dir

        self.raw_audio_dir = root / paths_cfg["raw_audio_dir"]
        self.processed_audio_dir = root / paths_cfg["processed_audio_dir"]
        self.features_h5_path = root / paths_cfg["features_h5"]
        self.models_dir = root / paths_cfg["models_dir"]
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Modèles classiques et quantiques
        self.svm_model_path = self.models_dir / "svm_rbf_model.joblib"
        self.qsvm_model_path = self.models_dir / "qsvm_model.joblib"
        self._svm_model: SVC | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    @staticmethod
    def _load_config(config_path: Path) -> PipelineConfig:
        """Charge la configuration globale à partir des fichiers YAML.

        Args:
            config_path: Chemin vers `config/paths.yaml`.

        Returns:
            PipelineConfig: configuration de base du pipeline.
        """

        import yaml  # import local pour garder les dépendances explicites

        root_dir = config_path.parent.parent.resolve()

        with config_path.open("r", encoding="utf-8") as f:
            paths_cfg: Dict[str, Any] = yaml.safe_load(f)

        audio_cfg_path = root_dir / "config" / "audio_params.yaml"
        quantum_cfg_path = root_dir / "config" / "quantum_params.yaml"

        with audio_cfg_path.open("r", encoding="utf-8") as f:
            audio_cfg: Dict[str, Any] = yaml.safe_load(f)

        with quantum_cfg_path.open("r", encoding="utf-8") as f:
            quantum_cfg: Dict[str, Any] = yaml.safe_load(f)

        return PipelineConfig(
            root_dir=root_dir, paths=paths_cfg, audio=audio_cfg, quantum=quantum_cfg
        )

    # ------------------------------------------------------------------
    # Données synthétiques (fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_synthetic_dataset(
        n_samples: int = 300, n_features: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Génère un petit dataset de classification binaire synthétique.

        Args:
            n_samples: Nombre d'échantillons à générer.
            n_features: Nombre de features par échantillon.

        Returns:
            Tuple (X, y) où X est un array (n_samples, n_features) et y un vecteur binaire.
        """

        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=4,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=42,
        )
        return X.astype(np.float32), y.astype(np.int64)

    # ------------------------------------------------------------------
    # Gestion des données audio réelles
    # ------------------------------------------------------------------
    def _build_audio_dataset_if_needed(self) -> bool:
        """Construit le dataset de features à partir des fichiers audio si besoin.

        Returns:
            True si un fichier HDF5 de features est disponible (créé ou existant).
        """

        if self.features_h5_path.exists():
            return True

        # Vérifie qu'il y a des fichiers audio bruts
        if not self.raw_audio_dir.exists():
            logger.warning(
                "[Audio] Répertoire raw_audio inexistant : %s", self.raw_audio_dir
            )
            return False

        # Paramètres audio pour le prétraitement
        audio_cfg = self.config.audio
        pre_params = PreprocessParams(
            sample_rate=int(audio_cfg.get("sample_rate", 22050)),
            segment_duration_seconds=float(
                audio_cfg.get("segment_duration_seconds", 8)
            ),
            silence_top_db=float(audio_cfg.get("silence_top_db", 30)),
        )

        pre = AudioPreprocessor(
            raw_dir=self.raw_audio_dir,
            processed_dir=self.processed_audio_dir,
            params=pre_params,
        )
        logger.info(
            "[Audio] Lancement du prétraitement audio (silence, segments, normalisation)…"
        )
        pre.run_full_pipeline()

        # Extraction de features + PCA
        from src.feature_extraction.core import FeatureExtractor, FeatureParams

        feat_params = FeatureParams(
            sample_rate=int(audio_cfg.get("sample_rate", 22050)),
            n_mfcc=int(audio_cfg.get("n_mfcc", 20)),
            hop_length=int(audio_cfg.get("hop_length", 512)),
            n_fft=int(audio_cfg.get("n_fft", 2048)),
            pca_components=int(self.config.quantum.get("pca_components", 8)),
        )
        extractor = FeatureExtractor(params=feat_params)

        logger.info("[Features] Extraction des features et PCA…")
        extractor.build_and_save_dataset(
            processed_root=self.processed_audio_dir,
            label_mapping=LABEL_MAPPING,
            output_h5=self.features_h5_path,
        )

        return self.features_h5_path.exists()

    def _load_features_from_h5(self) -> Tuple[np.ndarray, np.ndarray]:
        """Charge X et y depuis le fichier HDF5 de features."""

        if not self.features_h5_path.exists():
            raise FileNotFoundError(
                f"Fichier de features introuvable : {self.features_h5_path}"
            )

        with h5py.File(self.features_h5_path, "r") as f:
            X = np.array(f["X"], dtype=np.float32)
            y = np.array(f["y"], dtype=np.int64)
        return X, y

    # ------------------------------------------------------------------
    # Gestion des modèles
    # ------------------------------------------------------------------
    def _load_svm_or_raise(self) -> SVC:
        """Charge le modèle SVM RBF classique ou lève une erreur."""

        if self._svm_model is not None:
            return self._svm_model

        if not self.svm_model_path.exists():
            raise RuntimeError(
                f"Aucun modèle SVM trouvé à {self.svm_model_path}."
                " Veuillez lancer le mode train d'abord."
            )

        self._svm_model = joblib.load(self.svm_model_path)
        return self._svm_model

    # ------------------------------------------------------------------
    # API publique : train / evaluate / predict
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Entraîne les modèles SVM RBF classique et QSVM.

        Si des données audio sont disponibles, on utilise les features
        extraites. Sinon, on tombe sur un dataset synthétique.
        """

        have_audio_dataset = self._build_audio_dataset_if_needed()

        if have_audio_dataset:
            # Entraînement sur de vraies features audio
            X, y = self._load_features_from_h5()

            # Si les labels ne contiennent qu'une seule classe, basculer
            # vers le fallback synthétique pour éviter l'exception de
            # scikit-learn (qui exige au moins deux classes pour la fit).
            unique_labels = np.unique(y)
            if unique_labels.shape[0] < 2:
                logger.warning(
                    "[Train] Les labels des features contiennent une seule classe (%s). "
                    "Basculer sur le dataset synthétique.",
                    unique_labels,
                )
                have_audio_dataset = False
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

            # SVM RBF classique
            svm = SVC(kernel="rbf", probability=True, random_state=42)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_val)
            acc_svm = accuracy_score(y_val, y_pred)
            logger.info("[Train] Validation accuracy (SVM RBF audio) = %.3f", acc_svm)

            joblib.dump(svm, self.svm_model_path)
            self._svm_model = svm
            logger.info("[Train] Modèle SVM sauvegardé dans %s", self.svm_model_path)

            # QSVM
            try:
                qcfg = QuantumSVMConfig(
                    n_qubits=int(self.config.quantum.get("n_qubits", 8)),
                    shots=int(self.config.quantum.get("shots", 1024)),
                    backend_name=str(
                        self.config.quantum.get("backend", "aer_simulator")
                    ),
                    C=1.0,
                )
                qsvm = QuantumSVM(config=qcfg)

                # Pour limiter le temps de calcul, on peut sous-échantillonner
                max_q_train = min(50, X_train.shape[0])
                X_train_q = X_train[:max_q_train]
                y_train_q = y_train[:max_q_train]

                logger.info(
                    "[Train] Entraînement QSVM sur un sous-ensemble des données…"
                )
                qsvm.fit(X_train_q, y_train_q)
                y_pred_q = qsvm.predict(X_val)
                acc_qsvm = accuracy_score(y_val, y_pred_q)
                logger.info("[Train] Validation accuracy (QSVM audio) = %.3f", acc_qsvm)

                qsvm.save(self.qsvm_model_path)
                logger.info(
                    "[Train] Modèle QSVM sauvegardé dans %s", self.qsvm_model_path
                )
            except Exception as exc:  # pragma: no cover - dépend de Qiskit
                logger.error(
                    "[Train] QSVM non entraîné (erreur Qiskit ou backend) : %s", exc
                )
        else:
            # Fallback synthétique pour garder un pipeline exécutable
            logger.warning(
                "[Train] Aucune donnée audio trouvée, utilisation d'un dataset synthétique."
            )
            X, y = self._generate_synthetic_dataset()
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            svm = SVC(kernel="rbf", probability=True, random_state=42)
            svm.fit(X_train, y_train)

            y_pred = svm.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            logger.info("[Train] Validation accuracy (SVM RBF synthétique) = %.3f", acc)

            joblib.dump(svm, self.svm_model_path)
            self._svm_model = svm
            logger.info(
                "[Train] Modèle SVM synthétique sauvegardé dans %s", self.svm_model_path
            )

    def predict(self, audio_path: str) -> Dict[str, Any]:
        """Prédit le label pour un fichier audio avec SVM RBF et QSVM.

        Étapes :
        - Prétraitement de `audio_path` (suppression du silence, segmentation,
          normalisation RMS) via `AudioPreprocessor`.
        - Extraction des features + agrégation (mêmes paramètres que train).
        - Application du scaling + PCA appris pendant le train, à partir
          des statistiques sauvegardées dans le fichier HDF5.
        - Prédiction avec le SVM RBF classique et, si disponible, avec le
          modèle QSVM sauvegardé.

        Args:
            audio_path: Chemin du fichier audio à classifier.

        Returns:
            Dictionnaire contenant les prédictions SVM et QSVM.
        """

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")

        # Vérifie que le dataset de features et le modèle SVM existent
        if not self.features_h5_path.exists():
            raise RuntimeError(
                "Aucun dataset de features trouvé. Lancez d'abord le mode train pour "
                "construire les features et entraîner les modèles."
            )

        svm = self._load_svm_or_raise()

        # ------------------------------------------------------------------
        # 1) Prétraitement audio (en mémoire, segments sauvés sur disque)
        # ------------------------------------------------------------------
        audio_cfg = self.config.audio
        pre_params = PreprocessParams(
            sample_rate=int(audio_cfg.get("sample_rate", 22050)),
            segment_duration_seconds=float(
                audio_cfg.get("segment_duration_seconds", 8)
            ),
            silence_top_db=float(audio_cfg.get("silence_top_db", 30)),
        )

        pre = AudioPreprocessor(
            raw_dir=self.raw_audio_dir,
            processed_dir=self.processed_audio_dir,
            params=pre_params,
        )
        # On utilise un label spécial pour les segments de prédiction
        seg_paths = pre.process_file(audio_path, label="_predict")
        if not seg_paths:
            raise RuntimeError(
                "Aucun segment exploitable obtenu après prétraitement (tout silence ?)."
            )

        # ------------------------------------------------------------------
        # 2) Extraction de features sur les segments de prédiction
        # ------------------------------------------------------------------
        from src.feature_extraction.core import FeatureExtractor, FeatureParams

        feat_params = FeatureParams(
            sample_rate=int(audio_cfg.get("sample_rate", 22050)),
            n_mfcc=int(audio_cfg.get("n_mfcc", 20)),
            hop_length=int(audio_cfg.get("hop_length", 512)),
            n_fft=int(audio_cfg.get("n_fft", 2048)),
            pca_components=int(self.config.quantum.get("pca_components", 8)),
        )
        extractor = FeatureExtractor(params=feat_params)

        # On reconstruit les vecteurs de features brutes en réutilisant les
        # méthodes internes du FeatureExtractor.
        X_raw_list: list[np.ndarray] = []
        for p in seg_paths:
            raw_feats = extractor._extract_raw_features(p)  # type: ignore[attr-defined]
            vec = extractor._aggregate_features(raw_feats)  # type: ignore[attr-defined]
            X_raw_list.append(vec)

        X_raw = np.stack(X_raw_list, axis=0)

        # ------------------------------------------------------------------
        # 3) Scaling + PCA à partir des stats sauvegardées dans le HDF5
        # ------------------------------------------------------------------
        with h5py.File(self.features_h5_path, "r") as f:
            grp = f["scaling"]
            scaler_mean = np.array(grp["scaler_mean"], dtype=np.float32)
            scaler_scale = np.array(grp["scaler_scale"], dtype=np.float32)
            pca_components = np.array(grp["pca_components"], dtype=np.float32)
            pca_mean = np.array(grp["pca_mean"], dtype=np.float32)

        if X_raw.shape[1] != scaler_mean.shape[0]:
            raise RuntimeError(
                "Dimension des features de prédiction incompatible avec le scaler/PCA appris."
            )

        # Standardisation
        X_scaled = (X_raw - scaler_mean) / (scaler_scale + 1e-12)
        # Centrage pour la PCA
        X_centered = X_scaled - pca_mean
        # Projection
        X_reduced = X_centered @ pca_components.T

        # ------------------------------------------------------------------
        # 4) Prédictions SVM RBF et QSVM
        # ------------------------------------------------------------------
        svm_proba_segments = svm.predict_proba(X_reduced)
        svm_proba_mean = svm_proba_segments.mean(axis=0)
        svm_label = int(np.argmax(svm_proba_mean))

        qsvm_result: Dict[str, Any] | None = None
        if self.qsvm_model_path.exists():
            try:
                from src.models.quantum_svm import QuantumSVM as QSVMClass

                qsvm = QSVMClass.load(self.qsvm_model_path)
                q_proba_segments = qsvm.predict_proba(X_reduced)
                q_proba_mean = q_proba_segments.mean(axis=0)
                q_label = int(np.argmax(q_proba_mean))
                qsvm_result = {
                    "label": q_label,
                    "probabilities": q_proba_mean.tolist(),
                }
            except Exception as exc:  # pragma: no cover
                print(f"[Predict] Impossible d'utiliser le modèle QSVM : {exc}")

        return {
            "svm_rbf": {
                "label": svm_label,
                "probabilities": svm_proba_mean.tolist(),
            },
            "qsvm": qsvm_result,
            "n_segments": len(seg_paths),
        }

    def evaluate(self) -> Dict[str, float]:
        """Évalue les modèles SVM RBF et QSVM.

        - Si des features audio sont disponibles, on évalue sur celles-ci.
        - Sinon, on évalue uniquement le SVM sur un dataset synthétique.
        """

        metrics: Dict[str, float] = {}

        if self.features_h5_path.exists():
            X, y = self._load_features_from_h5()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # SVM RBF classique
            svm = self._load_svm_or_raise()
            y_pred_svm = svm.predict(X_test)
            acc_svm = accuracy_score(y_test, y_pred_svm)
            metrics["svm_rbf_accuracy"] = float(acc_svm)
            logger.info("[Eval] SVM RBF accuracy (audio) = %.3f", acc_svm)

            # QSVM s'il existe
            if self.qsvm_model_path.exists():
                try:
                    from src.models.quantum_svm import QuantumSVM

                    qsvm = QuantumSVM.load(self.qsvm_model_path)
                    y_pred_q = qsvm.predict(X_test)
                    acc_q = accuracy_score(y_test, y_pred_q)
                    metrics["qsvm_accuracy"] = float(acc_q)
                    logger.info("[Eval] QSVM accuracy (audio) = %.3f", acc_q)
                except Exception as exc:  # pragma: no cover
                    logger.error("[Eval] Impossible d'évaluer le QSVM : %s", exc)
        else:
            # Fallback synthétique
            svm = self._load_svm_or_raise()
            X_test, y_test = self._generate_synthetic_dataset(n_samples=200)
            y_pred = svm.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            metrics["svm_rbf_accuracy"] = float(acc)
            logger.info("[Eval] SVM RBF accuracy (synthétique) = %.3f", acc)

        return metrics
