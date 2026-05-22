# Explication du Workflow CI (Intégration Continue)

Ce fichier documente les actions réalisées par le workflow défini dans `ci.yml`. L'objectif de ce workflow est de s'assurer automatiquement que le code fonctionne correctement en exécutant les tests unitaires à chaque modification.

## 1. Déclencheurs (Triggers)
```yaml
on:
  push:
    branches: [ "main", "master" ]
  pull_request:
    branches: [ "main", "master" ]
```
Le workflow se déclenche automatiquement dès qu'un développeur pousse (push) du code sur la branche `main` (ou `master`), ou lorsqu'une Pull Request (demande de fusion) est ouverte vers cette branche.

## 2. Environnement
Le job `test` s'exécute sur `ubuntu-latest`, qui est un serveur virtuel Ubuntu frais fourni gratuitement par GitHub Actions.

## 3. Les différentes étapes (Steps)

1. **Checkout du code** (`actions/checkout@v4`) :
   Cette action officielle télécharge le code de votre dépôt GitHub sur la machine virtuelle pour pouvoir travailler dessus.

2. **Configuration de Python 3.10** (`actions/setup-python@v5`) :
   Installe la version 3.10 de Python sur la machine virtuelle, garantissant que les tests tournent dans un environnement identique à celui du développement.

3. **Installation des dépendances Python** :
   - Mise à jour de `pip`.
   - Installation de toutes les librairies listées dans `requirements.txt` (ex: fastapi, streamlit, opencv-python-headless, etc.).
   - Installation explicite de `pytest` (pour lancer les tests) et `httpx` (nécessaire pour simuler les requêtes vers l'API FastAPI pendant les tests).

4. **Exécution des tests (API et UI)** :
   Le script se place dans le dossier `dashboard` et lance la commande `pytest` sur les dossiers `api/tests/` et `ui/tests/`. 
   Si un seul test échoue, le workflow entier échoue et affiche une croix rouge (❌) sur GitHub. Si tout passe, il affiche une coche verte (✅), indiquant que le code est stable.
