### Partie 1 : Contexte et Objectifs (Intro)
*   Slide 0 : Page de garde 
    *   Titre du projet : "Preuve de concept : Segmentation médicale Zero-Shot avec MedSAM"
    *   Votre nom et le contexte (Test technique - DataSpace).
*   Slide 1 : Sommaire

*   Slide 2 : Le Contexte de la Mission (Chapitre 1)
    *   Objectif : Démontrer la supériorité d'un algorithme récent par rapport aux références historiques.
    *   Cas d'usage : Segmentation de lésions cutanées (première étape avant le diagnostic du mélanome).
*   Slide 3 : Le Jeu de données (ISIC 2018) (Chapitre 2)
    *   Présentation de l'ISIC (International Skin Imaging Collaboration).
    *   Rigueur clinique : Vérité terrain validée par le consensus géométrique de 5 experts.
*   Slide 4 : Répartition et Métadonnées
    *   Volume de données : Train (2 594), Validation (100), Test (1 000).
    *   *Optionnel* : Mention de l'utilisation des métadonnées (ISIC 2017) pour enrichir l'exploration dans le Dashboard.

### Partie 2 : L'État de l'art vs La Nouvelle Approche (Chapitre 3)
*   Slide 5 : Les Baselines (L'existant)
    *   **U-Net** : La norme historique en imagerie médicale (modèle entraîné *from scratch*).(Chapitre 3.1)
    *   **DeepLabV3+** : Modèle avancé avec architecture ResNet50 (entraîne sur le dataset ISIC).(Chapitre 3.2)
*   **Slide 6 : Le Nouvel Algorithme : MedSAM**
    *   Sorti en 2024, adaptation de *Segment Anything* (Meta) pour la médecine.(Chapitre 3.3)
    *   Modèle de Fondation : Pré-entraîné sur 1,4 million d'images médicales (15 modalités).
*   **Slide 7 : L'innovation : Le Prompting et le "Zero-Shot"**
    *   Explication du fonctionnement interactif : L'utilisateur fournit une boîte englobante (BBox) ou des clics.
    *   Le "Zero-Shot" : Capacité à segmenter la peau sans avoir été ré-entraîné (fine-tuné) spécifiquement sur notre dataset.
*   **Slide 8 : Protocole Expérimental**
    *   Comment les modèles ont été comparés (Même dataset de test).
    *   Présentation des métriques : **Dice Score** (superposition) et **IoU** (Jaccard).

### Partie 3 : Les Résultats (Slides 9 à 11)
*   **Slide 9 : Analyse Quantitative**
    *   Tableau des scores moyens sur le test set.
    *   Victoire de MedSAM (Dice : 0.919) contre DeepLabV3+ (0.885) et U-Net (0.842).
*   **Slide 10 : Analyse Qualitative (Visuelle)**
    *   *Mettre des captures d'écran des masques.*
    *   Explication visuelle : U-Net fragmente les masques flous, MedSAM isole la lésion en un seul bloc grâce à son attention globale.

### Partie 4 : Explicabilité et Feature Importance (Slides 12 à 15)
*   **Slide 11 : Pourquoi l'explicabilité (XAI) en médecine ?**
    *   Nécessité de comprendre la "boîte noire" pour instaurer la confiance clinique.
*   **Slide 12 : Feature Importance Globale (La morphologie)**
    *   Impact de la taille de la lésion sur les performances.
    *   Excellente stabilité sur les petites/moyennes lésions, difficulté sur les grandes lésions asymétriques.
*   **Slide 13 : Feature Importance Locale (Heatmaps)**
    *   Comment le Vision Transformer "réfléchit" (branché sur la dernière couche d'attention).
*   **Slide 14 : Les Biais de l'algorithme**
    *   Observation des phénomènes physiques : "Attention Sinks" (attention sur la peau saine), les cheveux comme distracteurs géométriques, et les biais de la grille de positionnement.

### Partie 5 : Limites, Améliorations et Dashboard (Slides 16 à 20)
*   **Slide 15 : Les Limites Identifiées**
    *   Dépendance exclusive aux boîtes englobantes (le mode "Plein Image" ou "clics seuls" perd en précision).
    *   Sensibilité aux artefacts de surface (poils, cheveux, reflets).
*   **Slide 16 : Pistes d'amélioration**
    *   Pré-traitement mathématique (algorithme DullRazor) pour gommer les cheveux.
    *   Mise en place d'un "Double Prompting" (Boîte + Clics correcteurs).
*   **Slide 17 : Démonstration du Dashboard (Le Livrable)**
    *   Analyse exploratoire (répartition des données, types de lésions).
    *   *Captures d'écran (ou live-demo si autorisée)* de la sélection d'une image et de la prédiction interactive.
    *   Prise en compte des normes d'accessibilité (WCAG) pour les graphiques.
*   **Slide 18 : Conclusion Générale**
    *   Bilan de la preuve de concept : Succès de l'approche "Foundation Model".
    *   L'avenir de la segmentation interactive en clinique.
*   **Slide 19 : Bibliographie et Sources**
    *   Mentions des articles (Nature Communications pour MedSAM, U-Net, etc.).
*   **Slide 20 : Questions / Réponses**