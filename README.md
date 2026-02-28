# Projet ML Retail — Analyse Comportementale Clientèle

## Description
Analyse du comportement client d'un e-commerce de cadeaux via Machine Learning.
Objectifs : segmentation, prédiction du churn, recommandations marketing.

## Installation

### 1. Cloner le projet
```bash
git clone https://github.com/imen767/Machine-Learning.git
cd projet_ml_retail
```

### 2. Créer l'environnement virtuel
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Structure du projet
- `data/raw/` : Données brutes originales
- `data/processed/` : Données nettoyées
- `data/train_test/` : Données splittées train/test
- `notebooks/` : Exploration et prototypage
- `src/` : Scripts Python production
- `models/` : Modèles sauvegardés
- `app/` : Application Flask
- `reports/` : Visualisations et rapports

## Utilisation
```bash
# Prétraitement
python src/preprocessing.py

# Entraînement
python src/train_model.py

# Prédiction
python src/predict.py
```