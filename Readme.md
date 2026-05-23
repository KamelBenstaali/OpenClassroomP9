# 🩺 MedSAM Segmentation Dashboard - OpenClassrooms Projet 9

Bienvenue dans le dépôt du projet de segmentation d'images médicales basé sur **MedSAM** (Segment Anything Model appliqué au domaine médical). 
Ce projet a été réalisé dans le cadre du parcours Data Scientist (Projet 9) et a pour objectif de concevoir, développer et déployer une solution complète (API backend + Dashboard frontend) d'aide au diagnostic médical via la segmentation d'images.

---

## 🔗 Liens de l'application en direct

- **L'application (Dashboard Streamlit)** : [https://gyfy85cyvcrq5fjqpp8kuv.streamlit.app](https://gyfy85cyvcrq5fjqpp8kuv.streamlit.app)
- **L'API (Hugging Face Spaces)** : [https://huggingface.co/spaces/S0l0kame/OpenClassroomP9](https://huggingface.co/spaces/S0l0kame/OpenClassroomP9)

---

## 🎯 Fonctionnalités Principales

Le projet est divisé en deux grandes composantes interactives hébergées dans le Cloud :

1. **Dashboard Interactif (Frontend)** :
   - **Analyse Exploratoire (EDA)** : Visualisation de la distribution des données médicales (résolutions, répartition des pathologies) et démonstration des techniques de "Data Augmentation" (floutage, égalisation, rotations).
   - **Moteur de Prédiction** : Interface interactive permettant à un utilisateur (ex: un médecin) de charger ou de sélectionner une image médicale, d'y dessiner une boîte de sélection (Bounding Box) ou de placer des points d'intérêt, et d'obtenir en temps réel un masque de segmentation précis des lésions ou organes.

2. **API de Deep Learning (Backend)** :
   - API robuste basée sur FastAPI.
   - Charge en mémoire le modèle "Foundation" **MedSAM** (ViT-B).
   - Reçoit l'image et les coordonnées (boîte/points) envoyées par le frontend.
   - Retourne un masque binarisé encodé en Base64.
   - Télécharge automatiquement ses poids (`medsam_vit_b.pth`) depuis le Hub Hugging Face lors du premier démarrage pour contourner les limites de stockage Git.

3. **Recherche & Modélisation (Notebooks)** :
   - Un dossier `Mes_Notebooks/` centralise tout le travail d'exploration et d'entraînement en amont du déploiement.
   - Vous y trouverez l'analyse approfondie du dataset ISIC 2018, les tests d'entraînement sur différents modèles classiques de segmentation (U-Net, DeepLabV3+ avec et sans Data Augmentation), ainsi que le "Proof of Concept" (POC) initial de **MedSAM** qui a mené au choix final de l'architecture déployée.

---

## 🏗️ Architecture Technique

- **Frontend** : [Streamlit](https://streamlit.io/) (Déployé sur Streamlit Community Cloud)
- **Backend / API** : [FastAPI](https://fastapi.tiangolo.com/) & [Uvicorn](https://www.uvicorn.org/) (Déployé sur Hugging Face Spaces via Docker)
- **Modèle de Deep Learning** : [MedSAM](https://github.com/bowang-lab/MedSAM) (Pytorch)
- **CI/CD** : [GitHub Actions](https://github.com/features/actions)
- **Tests Unitaires** : [Pytest](https://docs.pytest.org/) & `httpx`

---

## 🚀 Intégration et Déploiement Continus (CI/CD)

Le projet intègre des pipelines automatisés complets définis dans le dossier `.github/workflows/` :

- **CI (Tests Unitaires)** : À chaque `push` ou `pull_request` sur la branche principale, GitHub Actions lance automatiquement une batterie de tests avec Pytest sur l'API et l'UI pour garantir la non-régression du code.
- **CD (Déploiement Continu)** : À chaque `push` modifiant le code de l'API, le Dockerfile ou les dépendances, le code est automatiquement poussé vers **Hugging Face Spaces** en utilisant l'outil officiel `hf upload` afin d'éviter les problèmes liés à l'historique Git LFS.

> 📄 *Des fichiers explicatifs détaillés (`README_CI.md` et `README_CD.md`) se trouvent dans le dossier `.github/workflows/`.*

---

## 📂 Structure du projet

```text
Src/
│
├── dashboard/
│   ├── api/
│   │   ├── main.py              # Point d'entrée de l'API FastAPI
│   │   ├── tests/               # Scripts de tests unitaires (pytest) pour l'API
│   │   └── temp_images/         # Dossier temporaire pour la manipulation d'images
│   │
│   └── ui/
│       ├── app.py               # Application Streamlit (Frontend)
│       ├── tests/               # Scripts de tests unitaires pour l'UI
│       ├── data_example/        # Échantillons d'images médicales pour démo
│       └── data_eda/            # Données statistiques pour les graphiques EDA
│
├── Mes_Notebooks/               # Notebooks Jupyter (Recherche & Développement)
│   ├── Medical_datasets_analysis/ # Analyse exploratoire des datasets (EDA ISIC2018)
│   ├── Developping_models/        # Entraînements (UNet, DeepLabV3+, POC MedSAM, etc.)
│   └── Benchmarking_models.ipynb  # Comparaison des performances des modèles
│
├── .github/workflows/           # Pipelines CI/CD automatisés
│   ├── ci.yml
│   └── cd_huggingface.yml
│
├── Dockerfile                   # Environnement conteneurisé pour le déploiement de l'API
├── requirements.txt             # Dépendances Python communes
└── Readme.md                    # Documentation du projet (vous êtes ici)
```

---

## 🛠️ Installation et Exécution en local

Si vous souhaitez exécuter le projet sur votre propre machine :

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/VOTRE_NOM/OpenClassroomP9.git
   cd OpenClassroomP9/Src
   ```

2. **Créer un environnement virtuel et installer les dépendances** :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sous Windows : venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Lancer l'API FastAPI** :
   L'API se mettra automatiquement à télécharger les poids de MedSAM la première fois.
   ```bash
   cd dashboard/api
   uvicorn main:app --reload
   ```

4. **Lancer le Dashboard Streamlit** (dans un autre terminal) :
   ```bash
   cd Src/dashboard/ui
   streamlit run app.py
   ```

*(Pensez à modifier l'URL `API_URL` dans `app.py` vers `http://127.0.0.1:8000/predict/` si vous travaillez 100% en local).*

---

## 🤝 Accessibilité et Normes (WCAG)
L'interface Streamlit a été pensée pour respecter les bonnes pratiques. Les textes sont contrastés, les champs et graphiques disposent de descriptions (légendes), et l'interface reste lisible indépendamment du thème (clair ou sombre).
