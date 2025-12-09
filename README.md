# Projet QSVM pour classification audio folklorique

Du son folklorique aux qubits : préserver la culture par l'informatique quantique.

Ce projet implémente un pipeline complet pour la classification binaire de musique folklorique camerounaise (Gurna vs non-Gurna) en combinant traitement du signal audio, extraction de caractéristiques, SVM classique et QSVM avec encodage angulaire.

Voir `main.py` et `config/` pour le lancement du pipeline.

## Lancer le pipeline en local

Installation des dépendances :

```bash
pip install -r requirements.txt
```

Entraînement + évaluation :

```bash
python main.py --config config/paths.yaml --mode train
python main.py --config config/paths.yaml --mode evaluate
```

Prédiction sur un fichier audio :

```bash
python main.py --config config/paths.yaml --mode predict --audio chemin/mon_fichier.wav
```

## Lancer l'API FastAPI (local)

```bash
uvicorn api.app:app --reload
```

Puis ouvrir la documentation Swagger : `http://127.0.0.1:8000/docs`.

## Generer le rapport pdf avec LaTeX

```bash
pdflatex -interaction=nonstopmode -output-directory docs/rapport/ docs/rapport/rapport.tex 
```

## Déploiement avec Docker

Construire l'image et lancer l'API :

```bash
docker-compose up --build
```

L'API sera disponible sur `http://127.0.0.1:8000` (Swagger sur `/docs`).
