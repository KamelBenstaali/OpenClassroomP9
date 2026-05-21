# 🖥️ Interface Utilisateur (UI) - Dashboard de Segmentation

Ce répertoire contient l'interface graphique interactive développée avec **Streamlit** pour présenter les résultats de la preuve de concept (POC) du projet de segmentation de lésions cutanées (P9 - OpenClassrooms).

## 📋 Fonctionnalités (Conformes au cahier des charges)

Le dashboard est divisé en deux grandes sections :

1. **Analyse Exploratoire des Données (EDA)**
   - Visualisation de la répartition du jeu de données (Train/Validation/Test).
   - Utilisation d'une charte graphique accessible (**WCAG**), notamment la palette Okabe-Ito (colorblind-friendly).
   - Présentation interactive des transformations d'images (Data Augmentation et Pre-processing) avec des exemples concrets (Floutage Gaussien, Égalisation d'histogramme).

2. **Moteur de Prédiction**
   - Sélection des données d'entrée : possibilité de charger une image locale (`upload`) ou de choisir un exemple issu du jeu de données de test.
   - Appel à l'API backend (en cours d'intégration) pour générer les prédictions.
   - Affichage côte à côte de l'image originale et du masque de segmentation généré par le modèle (U-Net / DeepLabV3+).

## 🚀 Installation et Lancement

### 1. Prérequis
Assurez-vous d'avoir activé votre environnement virtuel principal, ou installez les dépendances nécessaires. Le package principal requis est `streamlit`.

```bash
pip install streamlit opencv-python Pillow matplotlib numpy
```
*(Vous pouvez aussi utiliser le fichier `requirements.txt` à la racine du dossier `Src`)*

### 2. Démarrer l'application locale
Depuis votre terminal, placez-vous dans le dossier `Src` puis lancez la commande suivante :

```bash
streamlit run dashboard/ui/app.py
```

L'interface s'ouvrira automatiquement dans votre navigateur web par défaut à l'adresse locale : `http://localhost:8501`.

## 📁 Structure
* `app.py` : Script principal contenant toute la logique d'affichage, la barre de navigation et l'interactivité utilisateur.

## 🧪 Tests Unitaires

Une suite de tests a été mise en place pour garantir le bon fonctionnement de la logique interne de l'interface utilisateur. Ces tests sont écrits avec `pytest` et se trouvent dans le dossier `tests/`.

- **`test_app.py`** : 
  - Teste la fonction de traitement d'images `apply_transformations` (utilisée dans l'onglet d'Analyse Exploratoire).
  - Vérifie que chaque transformation (floutage, égalisation, rotation, miroir) génère bien un tableau NumPy avec les bonnes dimensions, garantissant que les images modifiées seront correctement formatées pour l'affichage sans faire planter l'application Streamlit.
