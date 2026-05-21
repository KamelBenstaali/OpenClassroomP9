# ⚙️ Moteur de Prédiction (Backend API)

Ce répertoire a vocation à contenir l'API RESTful chargée de servir les modèles d'intelligence artificielle (Deep Learning) développés pour la segmentation sémantique des lésions cutanées.

## 🎯 Rôle de l'API

Afin de séparer la logique métier de l'interface utilisateur, l'architecture du projet s'appuie sur une API distincte. Ses principaux objectifs sont :
- **Chargement des modèles** : Instanciation en mémoire des modèles U-Net et/ou DeepLabV3+ entraînés.
- **Réception des requêtes** : Point d'entrée pour recevoir les images provenant de l'interface Streamlit (Dashboard UI).
- **Pré-traitement (Pre-processing)** : Standardisation, redimensionnement et normalisation des images d'entrée pour correspondre au format attendu par le modèle.
- **Inférence** : Génération des prédictions (masques de segmentation binaire).
- **Post-traitement** : Transformation du tenseur prédit en format exploitable par l'interface (ex: image binarisée encodée en base64).

## 🛠️ Stack Technologique Prévue

- **Framework Web** : `FastAPI` (recommandé pour ses excellentes performances et la documentation Swagger intégrée) ou `Flask`.
- **Machine Learning** : `TensorFlow`/`Keras` pour exécuter le graphe de prédiction.
- **Traitement d'Image** : `OpenCV`, `Pillow`, `NumPy`.

## 🌐 Routes de l'API (Conception initiale)

| Endpoint | Méthode HTTP | Description |
|---|---|---|
| `/` | `GET` | Vérification de l'état de santé du serveur (Health Check). |
| `/predict` | `POST` | Route principale. Accepte un fichier image multipart/form-data et retourne le masque de segmentation. |
| `/models` | `GET` | (Optionnel) Retourne la liste des modèles disponibles (ex: U-Net vs DeepLabV3+). |

## 🚀 Installation et Lancement (FastAPI)

### 1. Prérequis
Assurez-vous d'avoir installé les dépendances nécessaires au fonctionnement d'une API FastAPI (dans votre environnement virtuel) :
```bash
pip install fastapi uvicorn python-multipart
```

### 2. Démarrer l'API en local
Une fois le script `main.py` créé, vous pourrez démarrer le serveur Uvicorn avec la commande suivante (à exécuter depuis le dossier `Src`) :

```bash
uvicorn dashboard.api.main:app --reload
```
- L'API sera accessible à l'adresse : `http://127.0.0.1:8000`
- **Bonus FastAPI** : La documentation interactive (Swagger) pour tester vos routes sera générée automatiquement sur : `http://127.0.0.1:8000/docs`

## 🔜 Prochaines Étapes
1. Créer le script principal `main.py` pour initialiser l'application FastAPI.
2. Ajouter le code de chargement de votre modèle U-Net/DeepLab sauvegardé (`.keras` ou `.h5`).
3. Tester la route `/predict` directement via l'interface Swagger (`/docs`).
4. Connecter l'interface Streamlit (dossier `ui`) à cette API locale en utilisant la bibliothèque Python `requests`.

## 🧪 Tests Unitaires

L'API est couverte par des tests automatisés utilisant `pytest` et `TestClient` de FastAPI. Ils sont situés dans le dossier `tests/`.

- **`test_api.py`** vérifie trois comportements clés :
  - **`test_read_root`** : S'assure que le point d'entrée de santé (`/`) répond correctement (Code 200).
  - **`test_get_image_not_found`** : Vérifie que le gestionnaire d'erreurs attrape bien les demandes d'images inexistantes (`/images/{filename}`) et renvoie une erreur 404 explicite.
  - **`test_predict_segmentation_no_prompt`** : Teste la route de prédiction majeure (`/predict/`). En envoyant une image factice, on valide l'intégrité de l'analyse du payload, le traitement de l'image, et le format de retour (qui doit contenir le statut, le score de confiance, et le masque de segmentation encodé en Base64). Ce test est conçu pour ne pas bloquer les pipelines d'Intégration Continue (CI) même si les poids du modèle complet ne sont pas téléchargés sur le serveur d'intégration (tolérance au code 500 dans ce contexte précis).
