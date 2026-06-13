# Prédiction du Turnover des Employés

Projet de mémoire M2 — Capstone visant à prédire l'attrition (départ) des employés à partir de données RH, en combinant des modèles de Machine Learning et des techniques d'interprétabilité (SHAP).

## 🎯 Objectif

Construire et comparer plusieurs modèles de classification capables d'identifier les employés à risque de départ, afin d'aider les équipes RH à anticiper le turnover et à orienter leurs actions de rétention. Une attention particulière est portée à l'interprétabilité des modèles via SHAP, pour comprendre les facteurs qui influencent le plus les prédictions.

## 📊 Dataset

**IBM HR Analytics Employee Attrition Dataset**
- 1 470 employés, 35 variables
- Variable cible : `Attrition` (Yes/No), déséquilibrée (~16% de départs)
- Fichier : `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`

## 📁 Structure du projet

```
.
├── data/                 # Données brutes (CSV)
├── notebooks/            # Notebooks Jupyter (EDA, préparation, modélisation, SHAP)
├── figures/              # Graphiques et visualisations exportés
├── dashboard/            # Application de visualisation / restitution des résultats
├── requirements.txt      # Dépendances Python
└── README.md
```

## 📓 Notebooks

- **01_Setup_EDA.ipynb** : Setup de l'environnement et analyse exploratoire des données (EDA) — structure du dataset, valeurs manquantes, colonnes constantes, distribution de la variable cible, facteurs clés de départ.
- **02_Modelisation.ipynb** *(à venir)* : Préparation des données (encodage, split train/test, SMOTE) et étude comparative de modèles (Logistic Regression, Random Forest, XGBoost).

## 🧪 Méthodologie

1. **Préparation des données** : nettoyage, suppression des colonnes inutiles, encodage des variables catégorielles, split train/test stratifié (80/20)
2. **Gestion du déséquilibre** : SMOTE appliqué uniquement sur le jeu d'entraînement
3. **Modélisation** : comparaison de Logistic Regression, Random Forest et XGBoost
4. **Évaluation** : accuracy, precision, recall, F1-score, AUC-ROC, matrice de confusion (focus sur le recall/F1 de la classe minoritaire)
5. **Interprétabilité** : analyse SHAP pour identifier les variables les plus influentes

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

## 🔒 Reproductibilité

`random_state=42` est utilisé systématiquement pour garantir la reproductibilité des résultats.
