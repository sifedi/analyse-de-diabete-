import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Ignorer les avertissements obsolètes pour plus de clarté
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# Charger le dataset
data = pd.read_csv('C:\\Users\\USER\\Desktop\\projet data minig\\diabetes_selected_features_final.csv', sep=',')

# Définir les features et la cible
selected_features = ['Age', 'BMI', 'Glucose', 'SkinThickness']
X = data[selected_features]
y = data['Outcome']

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculer scale_pos_weight pour gérer le déséquilibre des classes
# Ratio négatifs/positifs dans y_train
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # Approx. 1.86 (65% 0, 35% 1)

# --- Étape 1 : Évaluation AVANT GridSearchCV (hyperparamètres par défaut) ---
print("\n=== Évaluation AVANT GridSearchCV (XGBoost par défaut) ===")

# Initialiser le modèle XGBoost avec scale_pos_weight
xgb_default = XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=scale_pos_weight)

# Entraîner le modèle
xgb_default.fit(X_train, y_train)

# Prédictions
y_pred_default = xgb_default.predict(X_test)

# Vérifier les classes prédites
print("Classes prédites (avant) :", np.unique(y_pred_default))

# Évaluation
print("Accuracy (avant) :", accuracy_score(y_test, y_pred_default))
print("Rapport de classification (avant) :\n", classification_report(y_test, y_pred_default))

# Matrice de confusion
cm_default = confusion_matrix(y_test, y_pred_default)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-diabétique', 'Diabétique'], yticklabels=['Non-diabétique', 'Diabétique'])
plt.title("Matrice de confusion (XGBoost - Avant GridSearchCV)")
plt.xlabel("Prédiction")
plt.ylabel("Vraie valeur")
plt.show()

# --- Étape 2 : Optimisation avec GridSearchCV (CV=4) ---
print("\n=== Optimisation avec GridSearchCV (CV=4) ===")

# Définir la grille d'hyperparamètres pour XGBoost
param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'min_child_weight': [1, 3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, scale_pos_weight]  # Tester sans et avec scale_pos_weight
}

# Initialiser le modèle XGBoost
xgb = XGBClassifier(eval_metric='logloss', random_state=42)

# Configurer GridSearchCV avec CV=4
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=4, scoring='f1', n_jobs=-1, verbose=1)

# Entraîner GridSearchCV
grid_search.fit(X_train, y_train)

# Afficher les meilleurs paramètres
print("\nMeilleurs paramètres après GridSearchCV (CV=4) :")
print(grid_search.best_params_)
print("Meilleur score F1 (validation, CV=4) :", grid_search.best_score_)

# --- Étape 3 : Évaluation APRÈS GridSearchCV ---
print("\n=== Évaluation APRÈS GridSearchCV (XGBoost optimisé) ===")

# Meilleur modèle
best_xgb = grid_search.best_estimator_
y_pred_best = best_xgb.predict(X_test)

# Vérifier les classes prédites
print("Classes prédites (après) :", np.unique(y_pred_best))

# Évaluation
print("Accuracy (après) :", accuracy_score(y_test, y_pred_best))
print("Rapport de classification (après) :\n", classification_report(y_test, y_pred_best))

# Matrice de confusion
cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-diabétique', 'Diabétique'], yticklabels=['Non-diabétique', 'Diabétique'])
plt.title("Matrice de confusion (XGBoost - Après GridSearchCV)")
plt.xlabel("Prédiction")
plt.ylabel("Vraie valeur")
plt.show()

# --- Étape 4 : Comparaison visuelle des F1-scores ---
# Extraire les F1-scores pour les classes 0 et 1 avant et après
report_default = classification_report(y_test, y_pred_default, output_dict=True)
report_best = classification_report(y_test, y_pred_best, output_dict=True)

# Gestion des clés manquantes
f1_class_0_default = report_default.get('0', {}).get('f1-score', 0.0)  # 0.0 si la classe 0 est absente
f1_class_0_best = report_best.get('0', {}).get('f1-score', 0.0)
f1_class_1_default = report_default.get('1', {}).get('f1-score', 0.0)
f1_class_1_best = report_best.get('1', {}).get('f1-score', 0.0)

# Créer un DataFrame pour visualisation
f1_scores = pd.DataFrame({
    'Classe 0 (Non-diabétique)': [f1_class_0_default, f1_class_0_best],
    'Classe 1 (Diabétique)': [f1_class_1_default, f1_class_1_best]
}, index=['Avant GridSearchCV', 'Après GridSearchCV'])

# Visualisation
f1_scores.plot(kind='bar', figsize=(8, 6))
plt.title("Comparaison des F1-scores avant et après GridSearchCV (CV=4)")
plt.ylabel("F1-score")
plt.xticks(rotation=0)
plt.legend(title="Classe")
plt.show()