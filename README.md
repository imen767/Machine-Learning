# 🛍️ Analyse Comportementale Clientèle Retail

Projet Machine Learning 
Analyse et prédiction du comportement des clients d'un e-commerce de cadeaux.

---

## 📋 Description

Ce projet implémente une chaîne complète de traitement en Data Science :

- **Exploration** : Analyse de la qualité et structure des données
- **Préparation** : Nettoyage, encoding et normalisation
- **Modélisation** : Clustering, Classification et Régression
- **Déploiement** : Application web Flask

---

## 📁 Structure du Projet
```
projet_ml_retail/
├── data/
│   ├── raw/          # Données brutes originales
│   ├── processed/    # Données nettoyées
│   └── train_test/   # Données splittées
├── notebooks/        # Notebooks Jupyter
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modelisation.ipynb
├── src/              # Scripts Python
│   ├── utils.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── models/           # Modèles sauvegardés
├── app/              # Application Flask
│   ├── app.py
│   └── templates/
│       └── index.html
├── reports/          # Rapports et visualisations
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Cloner le projet
```bash
git clone https://github.com/imen767/Machine-Learning.git
cd projet_ml_retail
```

### 2. Créer et activer l'environnement virtuel
```bash
# Créer
python -m venv venv

# Activer (Windows)
venv\Scripts\activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## 🚀 Guide d'utilisation

### Exécuter les notebooks
```bash
jupyter notebook
```
Ouvrir dans l'ordre :
1. `notebooks/01_exploration.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_modelisation.ipynb`

### Lancer l'application Flask
```bash
cd app
python app.py
```
Ouvrir le navigateur sur : `http://127.0.0.1:5000`

---

## 📊 Résultats

| Modèle | Métrique | Résultat |
|--------|----------|----------|
| **Clustering K-Means** | Segments | 4 groupes équilibrés |
| **Classification Random Forest** | Accuracy | 96.1% |
| **Régression Random Forest** | R² | 0.917 |

---

## 🔑 Features utilisées

- `Frequency` : Nombre de commandes
- `MonetaryAvg` : Montant moyen par commande
- `AvgDaysBetweenPurchases` : Jours entre achats
- `SatisfactionScore` : Score de satisfaction
- `RFMSegment` : Segment RFM client
- `LoyaltyLevel` : Niveau de fidélité
- `PreferredMonth` : Mois préféré d'achat

---

## 🛠️ Technologies utilisées

- **Python** 3.x
- **Pandas** / **NumPy** : Manipulation des données
- **Scikit-learn** : Modèles ML
- **Matplotlib** / **Seaborn** : Visualisations
- **Flask** : Application web
- **Joblib** : Sauvegarde des modèles