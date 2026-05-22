# Explication du Workflow CD (Déploiement Continu)

Ce fichier documente les actions réalisées par le workflow défini dans `cd_huggingface.yml`. L'objectif de ce workflow est de déployer automatiquement le code de l'API vers Hugging Face Spaces lorsqu'une mise à jour valide est effectuée.

## 1. Déclencheurs et Filtres
```yaml
on:
  push:
    branches: [ "main", "master" ]
    paths:
      - 'dashboard/api/**'
      - 'requirements.txt'
      - 'Dockerfile'
      - '.github/workflows/cd_huggingface.yml'
```
Contrairement à la CI, la CD ne se lance pas à chaque fois. Grâce au filtre `paths`, GitHub Actions ne déploiera l'API que si des fichiers modifiés concernent l'API (le dossier `dashboard/api/`), les dépendances (`requirements.txt`), la configuration du serveur (`Dockerfile`) ou le fichier de workflow lui-même. Si vous modifiez l'interface Streamlit (UI), l'API sur Hugging Face ne sera pas redéployée inutilement.

## 2. Environnement
Le job `deploy-to-hf-spaces` s'exécute lui aussi sur un serveur virtuel `ubuntu-latest`.

## 3. Les différentes étapes (Steps)

1. **Checkout du code** (`actions/checkout@v4`) :
   Télécharge le code du dépôt. On utilise l'argument `lfs: true` pour forcer le téléchargement des gros fichiers (comme le modèle `medsam_vit_b.pth` de 350 Mo) stockés via Git LFS.

2. **Configuration de Python 3.10** :
   Installe Python pour pouvoir utiliser les outils officiels de Hugging Face.

3. **Installation de Hugging Face Hub** :
   On installe la librairie Python `huggingface_hub` via `pip`. Cette librairie fournit l'interface en ligne de commande (CLI) nommée `hf`, qui est indispensable pour interagir proprement avec les serveurs de Hugging Face.

4. **Déploiement via `hf upload`** :
   C'est le cœur du déploiement. L'outil `hf upload` prend le contenu de votre projet (le code, le Dockerfile, le modèle .pth) et le téléverse directement dans votre "Space" Hugging Face (`S0l0kame/OpenClassroomP9`).
   - Il utilise le "Secret" GitHub `HF_TOKEN` pour s'authentifier de manière sécurisée sans que le mot de passe n'apparaisse dans le code.
   - **Avantage majeur** : En utilisant `hf upload` au lieu d'un simple `git push`, on évite tous les problèmes historiques liés à Git LFS. L'outil gère nativement le transfert des gros fichiers vers l'architecture de Hugging Face de façon fiable.
