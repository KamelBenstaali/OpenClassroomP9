# Plan prévisionnel : Segmentation Médicale "Zéro-Shot"

## Dataset retenu
Le dataset retenu est issu de l'**ISIC 2018 Challenge** (International Skin Imaging Collaboration) - *Task 1 : Lesion Boundary Segmentation*. La collaboration ISIC est une initiative mondiale prestigieuse, soutenue par la communauté scientifique internationale, dont le but est de standardiser l'imagerie dermatologique pour lutter contre la mortalité due au mélanome grâce à l'Intelligence Artificielle.

Ce dataset public, référence incontournable en dermatologie numérique, propose des milliers d'images de dermoscopie haute résolution de lésions cutanées variées (mélanomes, nævus, kératoses séborrhéiques) accompagnées de leurs masques de segmentation "vérité terrain". 

À souligner que les masques sont validés par 5 médecins dermatologues (chaque médecin déssine un masque ensuite le masque finale est la fusion des 5 masques). 
Voici la répartition des données de ce dataset: 
- **Données d'entraînement :** 2 594 images avec leurs masques de segmentation ground truth
- **Données de test :** 1 000 images avec leurs masques de segmentation ground truth 
- **Données de validation :** 100 images avec leurs masques de segmentation ground truth.


## Modèle envisagé
Le nouvel algorithme sélectionné est **MedSAM** (Segment Anything in Medical Images), publié début 2024.

MedSAM est un modèle de segmentation d'images médicales de pointe, basé sur l'architecture innovante du projet Segment Anything (SAM). Contrairement aux modèles traditionnels qui doivent être entraînés spécifiquement pour chaque type d'imagerie (radio, IRM, échographie), MedSAM est pré-entraîné sur un vaste ensemble de données couvrant plus d'un million d'images médicales provenant de 15 modalités différentes. Cette approche "Foundation Model" lui confère une capacité de généralisation exceptionnelle.

### En quoi il serait succeptible d'aporter la performance
L'innovation majeure de MedSAM réside dans son approche de segmentation interactive et guidée par le contexte. Au lieu de produire un masque global, il permet d'affiner la segmentation en temps réel grâce à des invites simples :

1. **Boîtes englobantes (Bounding Boxes) :** L'utilisateur trace un cadre autour de la zone suspecte, et MedSAM isole automatiquement la lésion.
2. **Points de guidage :** Des clics sur le bord ou le centre de la pathologie permettent de corriger précisément les contours.

Cette flexibilité est particulièrement précieuse en médecine, où les anatomies varient considérablement et où les frontières des tumeurs sont souvent floues. MedSAM est conçu pour fonctionner en "Zero-Shot", c'est-à-dire qu'il peut segmenter des pathologies inconnues sans nécessiter de ré-entraînement spécifique (fine-tuning), réduisant ainsi considérablement le temps et le coût de développement des outils d'aide au diagnostic.

Contrairement aux modèles classiques comme U-Net qui nécessitent d'être entraînés spécifiquement sur des images de peau pour être performants, MedSAM est un "Foundation Model" entraîné sur plus d'un million d'images médicales issues de diverses modalités. L'objectif de MedSAM est d'offrir une segmentation universelle et interactive : en fournissant simplement une boîte englobante (bounding box) ou un clic sur la lésion, le modèle génère le masque précis de la pathologie. Ce modèle a le potentiel d'apporter un gain de performance massif en termes de "Zero-Shot learning" (capacité à segmenter sans ré-entraînement) et d'éliminer le coût d'annotation manuelle en clinique.

### Objectif de l'algorithme
Notre algorithme (MedSAM) a pour seule mission de **détecter exactement où se trouve la maladie** et de la séparer de la peau saine. C’est la toute première étape indispensable avant qu’un autre algorithme (ou un médecin) ne prenne le relais pour analyser la forme de la tache et dire si elle est cancéreuse !

## Références bibliographiques
### Références sur notre model MedSAM
1. **Article de recherche principal :** Ma, J., He, Y., Li, F. et al. *Segment Anything in Medical Images* (MedSAM). Nature Communications 15, 654 (2024). [Lien](https://www.nature.com/articles/s41467-024-44824-z). Cet article montre un benchmark entre medsam et les modeles de l'état de l'art.
2. **Article fondamental (Inspiration) :** Kirillov, A., Mintun, E., Ravi, N. et al. *Segment Anything* (SAM). arXiv:2304.02643 (2023). [Lien](https://arxiv.org/abs/2304.02643). Article introduisant le model **Segment Anything**.
3. **Dépôt officiel du modèle :** Dépot Github du projet MedSAM. [Lien](https://github.com/bowang-lab/MedSAM).

### Références sur notre baseline état d'art model U-Net et deeplabV3+ dans la segmentation d'images médicales
1. **Article de recherche :** Ma, J., He, Y., Li, F. et al. *Segment Anything in Medical Images* (MedSAM). Nature Communications 15, 654 (2024). [Lien](https://www.nature.com/articles/s41467-024-44824-z). *Cet article démontre, via un benchmark massif sur 15 modalités d'imagerie, comment MedSAM surpasse les modèles existants en "Zéro-Shot".*

2. **Article de conférence :** Effect of Segmentation on Skin Lesion Classification Using Multi-Scale Convolution Neural Networks, c'est un article qui décrit une étude faite sur l'impact de la segmentation sur la classification des lésions cutanées en utilisant deeplabV3+.
[Lien](https://ieeexplore.ieee.org/abstract/document/10958450)

### Références officielles sur les modeles de baseline
1. **Article de référence (SOTA "Baseline" Historique) :** Ronneberger, O., Fischer, P., & Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI (2015). [Lien](https://arxiv.org/abs/1505.04597). *L'article fondateur de U-Net, qui représente l'état de l'art classique (la baseline à battre) pour les modèles entraînés sur-mesure en imagerie médicale.*

2. **Article de référence (SOTA "Baseline" Historique) :** 
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff,
and Hartwig Adam. Encoder-decoder with atrous separable convolution
for semantic image segmentation.[Lien](https://arxiv.org/pdf/1802.02611).
*L'article fondateur de DeeplabV3+ qui a révolutionné la segmentation d'images.

## Explication de votre démarche de test du nouvel algorithme (votre preuve de concept)
La démarche de cette preuve de concept (POC) vise à démontrer la supériorité de MedSAM en matière de flexibilité et de généralisation (Zero-Shot) face aux architectures sur-mesure classiques.
*   **La Baseline :** Je vais utiliser les modeles de l'architecture **U-Net** et **deeplabV3+** (la norme historique en imagerie médicale). Je démontrerai que MedSAM est plus performant que ces deux modeles sur le dataset ISIC 2018.

*   **La Méthode (MedSAM) :** Je vais utiliser MedSAM de manière interactive, en lui fournissant des *prompts* (points de guidage) pour isoler les lésions cutanées.
*   **Comparaison :** J'évaluerai la performance des deux approches en utilisant le **Score Dice (DSC)** et **Jaccard Index (IoU)**, qui sont les métriques de référence en segmentation d'images médicales.

Le POC prendra la forme d'un Dashboard Streamlit permettant à l'utilisateur de cliquer sur une image de peau pour voir MedSAM détourer le mélanome instantanément, validant ainsi la puissance de l'algorithme récent sans aucun "fine-tuning".
