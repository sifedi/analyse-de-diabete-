# 🧠 Prédiction du Diabète avec l'Apprentissage Automatique

Ce projet a pour objectif de prédire la présence de diabète chez les patients en utilisant des algorithmes d'apprentissage automatique, basés sur le célèbre jeu de données **PIMA Indians Diabetes**.

## 📌 Objectifs
- Nettoyer et pré-traiter des données médicales
- Sélectionner les variables pertinentes
- Comparer les performances de deux modèles : Random Forest et XGBoost
- Optimiser les résultats pour un diagnostic précoce du diabète

## 🗃️ Jeu de données
Le jeu de données utilisé est disponible sur [Kaggle - PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

Il comprend :
- 768 observations
- 9 variables médicales
- Une variable cible : `Outcome` (1 = diabétique, 0 = non-diabétique)

## 🔬 Méthodologie
- Imputation des valeurs manquantes avec **KNN Imputer**
- Détection et traitement des valeurs aberrantes
- Standardisation des variables
- Sélection des features importantes via Random Forest
- Entraînement des modèles Random Forest et XGBoost
- Évaluation avec le **F1-score**

## 📊 Résultats
| Modèle        | F1-score |
|---------------|----------|
| Random Forest | 0.80     |
| XGBoost       | 0.82     |

## 🧭 Perspectives
- Ajouter d'autres variables (habitudes, antécédents)
- Tester le modèle sur des données réelles
- Déployer une API ou une interface utilisateur

## 🛠️ Installation
```bash
pip install -r requirements.txt
