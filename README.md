# 🌾 Wheat Seeds Segmentation — CAH

Application de **segmentation des graines de blé** basée sur la **Classification Hiérarchique Ascendante (CAH)** et développée avec **Python** et **Streamlit**.

🔗 **Application en ligne :**
https://wheatseedscah-2u5nu2jzskk2ckpcxhkfcd.streamlit.app/

---

## 📌 Description du projet

Ce projet consiste à segmenter des graines de blé en différents groupes selon leurs **caractéristiques morphologiques**.

L'objectif est d'identifier des groupes de graines présentant des caractéristiques similaires à partir de plusieurs mesures physiques et morphologiques.

La méthode utilisée est la **Classification Hiérarchique Ascendante (CAH)**, une méthode de clustering non supervisé permettant de regrouper les observations selon leur proximité.

L'application permet ensuite à un utilisateur de saisir les caractéristiques d'une nouvelle graine et d'obtenir automatiquement le groupe auquel elle est la plus proche.

---

## 🎯 Objectifs

Les principaux objectifs du projet sont :

* Explorer les caractéristiques morphologiques des graines de blé.
* Identifier des groupes homogènes de graines.
* Appliquer une méthode de clustering non supervisé.
* Interpréter les profils moyens des groupes obtenus.
* Développer une fonction permettant d'affecter une nouvelle graine à un groupe.
* Déployer le modèle sous forme d'une application web interactive.

---

## 📊 Variables utilisées

Le modèle utilise **7 caractéristiques morphologiques** :

| Variable                  | Description                     |
| ------------------------- | ------------------------------- |
| `area A`                  | Aire de la graine               |
| `perimeter`               | Périmètre de la graine          |
| `compactness`             | Compacité de la graine          |
| `length of kernel`        | Longueur de la graine           |
| `width of kernel`         | Largeur de la graine            |
| `asymmetry coefficient`   | Coefficient d'asymétrie         |
| `length of kernel groove` | Longueur du sillon de la graine |

Ces variables permettent de représenter la forme et les dimensions des graines.

---

## 🧠 Méthodologie

### 1. Préparation des données

Les données sont préparées afin de pouvoir appliquer la méthode de clustering sur les caractéristiques morphologiques sélectionnées.

### 2. Classification Hiérarchique Ascendante

La CAH est utilisée pour construire une partition des graines en **2 groupes**.

Le modèle utilisé repose sur :

* Nombre de clusters : `2`
* Méthode de liaison : `average`

La CAH permet de construire progressivement des groupes en fusionnant les observations ou groupes les plus proches.

### 3. Profilage des groupes

Après l'application de la CAH, les caractéristiques moyennes de chaque groupe sont calculées.

Ces profils moyens permettent d'interpréter les différences morphologiques entre les groupes.

### 4. Affectation d'une nouvelle graine

La CAH utilisée dans le projet ne fournit pas directement une méthode `predict()` pour une nouvelle observation.

Pour résoudre ce problème, le projet utilise le **profil moyen de chaque groupe comme centre représentatif**.

Pour une nouvelle graine :

1. Les caractéristiques saisies sont récupérées.
2. La distance entre la nouvelle graine et chaque profil moyen est calculée.
3. La graine est affectée au groupe dont le profil est le plus proche.

La distance euclidienne est utilisée :

```text
d(x, c) = √Σ(xᵢ - cᵢ)²
```

où :

* `x` représente la nouvelle graine ;
* `c` représente le profil moyen d'un groupe.

---

## 🖥️ Application Streamlit

L'application web permet à l'utilisateur de renseigner les sept caractéristiques d'une graine :

```text
Area A
Perimeter
Compactness
Length of kernel
Width of kernel
Asymmetry coefficient
Length of kernel groove
```

Après validation, l'application retourne :

* la classe prédite ;
* le profil correspondant ;
* la distance au profil moyen du groupe sélectionné.

### 🚀 Démo

👉 **Tester l'application :**

https://wheatseedscah-2u5nu2jzskk2ckpcxhkfcd.streamlit.app/

---

## 📁 Structure du projet

```text
Wheat_Seeds_CAH/
│
├── app.py
├── modele_cah.joblib
├── centres_cah.csv
├── features.json
├── valeurs_defaut.json
├── exemples.json
├── noms_classes.json
├── requirements.txt
└── README.md
```

### Description des fichiers

| Fichier               | Rôle                                          |
| --------------------- | --------------------------------------------- |
| `app.py`              | Application Streamlit                         |
| `modele_cah.joblib`   | Modèle CAH entraîné                           |
| `centres_cah.csv`     | Profils moyens des groupes                    |
| `features.json`       | Liste des variables utilisées                 |
| `valeurs_defaut.json` | Valeurs utilisées par défaut dans l'interface |
| `exemples.json`       | Exemples proposés dans l'application          |
| `noms_classes.json`   | Noms associés aux groupes                     |
| `requirements.txt`    | Dépendances Python                            |
| `README.md`           | Documentation du projet                       |

---

## 🛠️ Technologies utilisées

### Langage

* Python

### Data Science

* NumPy
* Pandas
* Scikit-learn
* Joblib

### Machine Learning

* Classification Hiérarchique Ascendante (CAH)
* Clustering non supervisé
* Distance euclidienne

### Déploiement

* Streamlit
* GitHub
* Streamlit Community Cloud

---

## ⚙️ Installation en local

Cloner le dépôt :

```bash
git clone https://github.com/VOTRE_USERNAME/Wheat_Seeds_CAH.git
```

Se placer dans le projet :

```bash
cd Wheat_Seeds_CAH
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer l'application :

```bash
streamlit run app.py
```

L'application sera ensuite accessible localement à l'adresse :

```text
http://localhost:8501
```

---

## 📈 Principe général

```text
Données morphologiques
        │
        ▼
Préparation des données
        │
        ▼
Classification Hiérarchique
Ascendante (CAH)
        │
        ▼
     2 groupes
        │
        ▼
Calcul des profils moyens
        │
        ▼
Nouvelle graine
        │
        ▼
Calcul des distances
        │
        ▼
Groupe le plus proche
        │
        ▼
Classe prédite
```

---

## 🔎 Limites

La méthode d'affectation d'une nouvelle graine repose sur la distance entre celle-ci et les **profils moyens des groupes** obtenus avec la CAH.

Cette approche constitue une stratégie simple et interprétable pour utiliser un modèle de clustering hiérarchique avec de nouvelles observations.

Elle ne correspond pas à une fonction `predict()` native de `AgglomerativeClustering`.

---

## 👨‍💻 Auteur

**Abou Birane BARO**

Data Scientist | Data Analyst | Machine Learning

📍 Dakar, Sénégal

---

## 📜 Licence

Ce projet est réalisé dans un cadre académique et de portfolio Data Science.
