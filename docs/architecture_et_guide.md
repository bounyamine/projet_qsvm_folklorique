# Projet QSVM pour classification audio folklorique

Du son folklorique aux qubits : préserver la culture par l'informatique quantique.

Ce document résume l’architecture du projet et le flux de données de bout en bout.

## 1. Vue d’ensemble du pipeline

1. **Prétraitement audio** (`src/data_pipeline`)

   - Entrée : `data/raw_audio/<label>/*.wav|*.mp3` (ex. `gurna`, `non_gurna`).
   - Sortie : segments WAV normalisés dans `data/processed_audio/<label>/*.wav`.

2. **Extraction de features** (`src/feature_extraction`)

   - Entrée : segments prétraités.
   - Sortie : matrice de features réduites (StandardScaler + PCA) stockée dans `results/features/extracted_features.h5`.

3. **Cœur quantique** (`src/quantum_module`)

   - Encodage angulaire des features en rotations `RY`.
   - Calcul du noyau quantique via estimation de fidélité.
   - Construction de matrices de Gram quantiques.

4. **Modèles ML** (`src/models`)

   - `QuantumSVM` : wrapper autour de `sklearn.svm.SVC(kernel="precomputed")`.
   - Baseline SVM RBF classique (`sklearn.svm.SVC(kernel="rbf")`).

5. **Pipeline orchestration** (`src/pipeline`)

   - Classe `AudioQSVMpipeline` gère :
     - `train()` : prétraitement, features, entraînement SVM + QSVM.
     - `predict(audio_path)` : prédiction fichier audio via SVM + QSVM.
     - `evaluate()` : évaluation et comparaison.

6. **API REST** (`api/`)

   - FastAPI expose :
     - `POST /predict` : prédiction sur upload audio.
     - `GET /model_info` : état et métriques.
     - `POST /train` : réentraînement (optionnellement protégé).

7. **Évaluation & visualisation** (`src/evaluation`)
   - `ModelEvaluator` calcule accuracy, precision, recall, F1, ROC-AUC.
   - Génère matrices de confusion et courbes ROC dans `results/evaluations/`.

## 2. Flux détaillé d’entraînement

1. Lancement :

```bash
python main.py --config config/paths.yaml --mode train
```

2. Chargement de la configuration :

   - `config/paths.yaml` : chemins de données, résultats, modèles.
   - `config/audio_params.yaml` : paramètres audio (sr, hop_length…).
   - `config/quantum_params.yaml` : n_qubits, shots, backend, pca_components.

3. Prétraitement audio :

   - `AudioPreprocessor.run_full_pipeline()` parcourt `data/raw_audio/`.
   - Supprime les silences (seuil `silence_top_db`).
   - Découpe en segments de durée `segment_duration_seconds`.
   - Normalise le RMS vers une cible fixe.

4. Extraction de features :

   - `FeatureExtractor.build_and_save_dataset()` construit `X, y, file_ids`.
   - Features extraites via _librosa_ : MFCC, chroma, spectral contrast, tonnetz, ZCR, RMS, centroid, bandwidth, rolloff, tempo.
   - Agrégation par moyennes/écarts-types temporels.
   - Standardisation + PCA → `X_reduced`.

5. Entraînement SVM RBF :

   - Split train/validation (`train_test_split`, stratifié).
   - Entraînement `SVC(kernel="rbf", probability=True)`.
   - Sauvegarde sous `models/svm_rbf_model.joblib`.

6. Entraînement QSVM :
   - Chargement des paramètres quantiques (`n_qubits`, `shots`, `backend`).
   - Sous-échantillonnage éventuel du train pour réduire le coût (max 50 points).
   - Calcul de la matrice de Gram quantique.
   - Entraînement `SVC(kernel="precomputed")`.
   - Sauvegarde du modèle QSVM sous `models/qsvm_model.joblib`.

## 3. Flux de prédiction

1. L’API /predict ou `AudioQSVMpipeline.predict(audio_path)` :

   - Vérifie qu’un dataset de features et un modèle SVM existent.
   - Prétraite le fichier audio (silence, segmentation, normalisation) vers `data/processed_audio/_predict/`.
   - Extrait les features pour chaque segment, puis applique le scaler + PCA appris.
   - Prédit pour chaque segment avec SVM (et QSVM si disponible).
   - Agrège par moyenne des probabilités sur les segments.

2. Sortie :
   - Label prédit (0/1) et probabilités par classe pour SVM RBF et QSVM.
   - Nombre de segments utilisés (`n_segments`).

## 4. Évaluation et visualisation

L’évaluation se fait soit via :

- `AudioQSVMpipeline.evaluate()` pour des métriques rapides, soit
- `src/evaluation/ModelEvaluator` pour générer des figures.

Exemple d’utilisation de `ModelEvaluator` dans un notebook :

```python
from pathlib import Path
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from src.evaluation import ModelEvaluator
from src.pipeline import AudioQSVMpipeline

pipeline = AudioQSVMpipeline(config_path="config/paths.yaml")

with h5py.File("results/features/extracted_features.h5", "r") as f:
    X = np.array(f["X"])
    y = np.array(f["y"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

svm = pipeline._load_svm_or_raise()
y_pred = svm.predict(X_test)
y_proba = svm.predict_proba(X_test)

results_dir = Path("results/evaluations")
eval = ModelEvaluator(results_dir=results_dir)
res = eval.evaluate_model(y_true=y_test, y_pred=y_pred, y_proba=y_proba, model_name="svm_rbf")
print(res.metrics)
```

Les figures sont stockées dans :

- `results/evaluations/confusion_matrices/`
- `results/evaluations/roc_curves/`

## 5. API REST et déploiement

- Lancer en local :

```bash
uvicorn api.app:app --reload
```

- Documentation Swagger : `http://127.0.0.1:8000/docs`.

- Déploiement Docker :

```bash
docker-compose up --build
```

Les volumes montés permettent de persister `data/`, `models/`, `results/` sur l’hôte.

## 6. Notes pour un rapport scientifique

- **Innovation** : application d’un QSVM à la classification de musique folklorique camerounaise.
- **Comparaison** : baseline SVM RBF vs QSVM, avec métriques et visualisations.
- **Reproductibilité** :
  - Configuration externalisée (`config/*.yaml`).
  - Script d’entrée unique (`main.py`).
  - API documentée (FastAPI + Swagger).
  - Conteneurisation Docker.
