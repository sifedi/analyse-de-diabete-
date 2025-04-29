import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données pré-traitées (ou après sélection des features)
# On suppose que tu as déjà fait le pré-traitement et la sélection des features
data = pd.read_csv('C:\\Users\\USER\\Desktop\\projet data minig\\diabetes_selected_features_final.csv',sep=',')
# Si tu n'as pas encore les données pré-traitées, utilise le code de pré-traitement précédent

# Définir les features et la cible (basé sur les features sélectionnées)
selected_features = ['Age','BMI','Glucose','SkinThickness']
X = data[selected_features]
y = data['Outcome']

# Étape 1 : Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Étape 2 : Entraîner un Random Forest de base
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# Étape 3 : Prédictions sur l'ensemble de test
y_pred = rf.predict(X_test)

# Étape 4 : Évaluation du modèle
print("Évaluation du modèle Random Forest :")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-diabétique', 'Diabétique'], yticklabels=['Non-diabétique', 'Diabétique'])
plt.title("Matrice de confusion")
plt.xlabel("Prédiction")
plt.ylabel("Vraie valeur")
plt.show()

# Étape 5 : Importance des features
importances = rf.feature_importances_
feature_importance = pd.Series(importances, index=selected_features).sort_values(ascending=False)
print("\nImportance des features :")
print(feature_importance)

# Visualisation de l'importance des features
plt.figure(figsize=(8, 6))
feature_importance.plot(kind='bar')
plt.title("Importance des features (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# Étape 6 : Validation croisée pour évaluer la robustesse
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='f1')
print("\nScores F1 en validation croisée (5 folds) :")
print(cv_scores)
print("Moyenne F1 :", cv_scores.mean())
print("Écart-type F1 :", cv_scores.std())

# Étape 7 : Optimisation des hyperparamètres avec GridSearchCV (facultatif)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Meilleurs paramètres et performance
print("\nMeilleurs paramètres après GridSearchCV :")
print(grid_search.best_params_)
print("Meilleur score F1 (validation) :", grid_search.best_score_)

# Évaluation du meilleur modèle sur l'ensemble de test
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)
print("\nPerformance du meilleur modèle sur l'ensemble de test :")
print(classification_report(y_test, y_pred_best))