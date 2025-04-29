import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# Étape 1 : Charger et structurer les données
data_split = pd.read_csv("C:\\Users\\USER\\Desktop\\projet data minig\\diabetes_apres1traitement.csv", sep=',')  


# Définir les features et la cible
features = ['Pregnanciess', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']
X = data_split[features]
y = data_split['Outcome']

# Étape 1 : Analyse de corrélation
print("Étape 1 : Analyse de corrélation")
correlation_matrix = data_split.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Matrice de corrélation")
plt.show()

# Identifier les variables les plus corrélées avec Outcome
correlation_with_outcome = correlation_matrix['Outcome'].drop('Outcome').abs().sort_values(ascending=False)
print("Corrélations avec Outcome (valeurs absolues) :")
print(correlation_with_outcome)

# Identifier les paires de variables fortement corrélées entre elles (> 0.7)
threshold = 0.7
correlated_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
print("Paires de variables fortement corrélées (> 0.7) :")
print(correlated_pairs)

# Étape 2 : Importance des features avec Random Forest
print("\nÉtape 2 : Importance des features avec Random Forest")
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
feature_importance_rf = pd.Series(importances, index=features).sort_values(ascending=False)
print("Importance des features (Random Forest) :")
print(feature_importance_rf)

# Visualisation de l'importance des features
plt.figure(figsize=(8, 6))
feature_importance_rf.plot(kind='bar')
plt.title("Importance des features (Random Forest)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# Étape 3 : SelectKBest avec test statistique (ANOVA F-test)
print("\nÉtape 3 : SelectKBest avec test statistique (ANOVA F-test)")
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)
scores = selector.scores_
feature_scores = pd.Series(scores, index=features).sort_values(ascending=False)
print("Scores des features (SelectKBest) :")
print(feature_scores)

# Visualisation des scores
plt.figure(figsize=(8, 6))
feature_scores.plot(kind='bar')
plt.title("Scores des features (SelectKBest)")
plt.xlabel("Features")
plt.ylabel("Score")
plt.show()

# Étape 4 : Sélection finale des features
# On combine les résultats pour choisir les meilleures features
# Par exemple, on garde les features qui apparaissent comme importantes dans les deux méthodes
top_features_rf = feature_importance_rf.head(5).index.tolist()  # Top 5 de Random Forest
top_features_selectkbest = feature_scores.head(5).index.tolist()  # Top 5 de SelectKBest
selected_features = list(set(top_features_rf).intersection(top_features_selectkbest))
print("\nFeatures sélectionnées (intersection des deux méthodes) :")
print(selected_features)

# Créer un nouveau DataFrame avec les features sélectionnées
data_selected = data_split[selected_features + ['Outcome']]
data_selected.to_csv('C:\\Users\\USER\\Desktop\\projet data minig\\diabetes_selected_features_final.csv', index=False)
print("Données avec features sélectionnées sauvegardées dans 'diabetes_selected_features4.csv'")