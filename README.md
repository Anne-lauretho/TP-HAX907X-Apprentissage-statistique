# TP-HAX907X-Apprentissage-statistique

Bienvenue dans notre dépôt pour le TP "Support Vector Machine" pour le cours HAX815X de l'année universitaire 2025-2026.
Les auteurs sont :

- STETSUN Kateryna (kateryna.stetsun@etu.umontpellier.fr)
- THOMAS Anne-Laure (anne-laure.thomas@etu.umontpellier.fr)

## Description
Ce dépôt contient le compte-rendu et les scripts Python pour ce TP.

Le TP couvre :
- Classification sur un dataset jouet : deux gaussiennes.
- Classification sur le dataset Iris.
- Classification de visages avec la base LFW (Labeled Faces in the Wild).
- Étude de l’influence des paramètres (C, noyau, nuisance, PCA).

## Structure
- Compte_rendu_STETSUN_THOMAS.qmd : le fichier Quarto contenant le compte-rendu du TP.
- À compléter. 

## Installation des dépendances et génération du rapport

Pour exécuter ce projet et générer le rapport HTML à partir du fichier `.qmd`, suivez les étapes ci-dessous :

1. **Créer un environnement virtuel** (recommandé) :

```bash
python -m venv venv
```

2. **Activer l’environnement virtuel** :

- Sur Windows :
```bash
venv\Scripts\activate
```

- Sur Linux / Mac :
```bash
source venv/bin/activate
```

3. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

4. **Générer le rapport HTML à partir du fichier .qmd** :

Pour générer le rapport HTML à partir des fichiers `.qmd`, il est nécessaire d'installer **Quarto** sur votre ordinateur. Suivez les étapes ci-dessous :

- Rendez-vous sur [le site officiel de Quarto](https://quarto.org/docs/get-started/).
- Téléchargez la version correspondant à votre système d'exploitation.
- Installez Quarto en suivant les instructions de l'installateur.
- Vérifiez l'installation en ouvrant un terminal ou l’invite de commande et en tapant :
```bash
quarto --version
```
- Exécuter:
```bash
quarto render Compte_rendu_STETSUN_THOMAS.qmd --to html
```

Le fichier HTML généré contiendra toutes les parties de code Python exécutables ainsi que les graphiques et analyses.
Veuillez faire preuve de patience : certaines parties du code peuvent prendre plus de temps à s'exécuter que d'autres (comme par exemple la cellule 4). La conversion complète du rapport peut donc nécessiter quelques moments selon la complexité des calculs.

Voici un schéma de l'architecture de notre projet, détaillant l'emplacement de chaque dossier et fichier : (À modifier)

```TP_SVM
    ├── TP-HAX907X-Apprentissage-statistique/
    │    ├── Licence
    │    ├── Compte_rendu_STETSUN_THOMAS.qmd
    │    └── README.md
```