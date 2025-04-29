import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
# Étape 1 : Charger et structurer les données
data_split = pd.read_csv('C:\\Users\\USER\\Desktop\\projet data minig\\diabetes.csv', sep=';')  

data_split = data_split.astype(float)

# Étape 1 : Gestion des zéros impossibles (valeurs manquantes déguisées)
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_with_zeros:
    data_split[column] = data_split[column].replace(0, data_split[column].median())

# Étape 2 : Gestion des valeurs aberrantes
# 2.1 : Règles médicales pour identifier les valeurs impossibles
rules = {
    'Glucose': (40, 300),
    'BloodPressure': (40, 150),
    'SkinThickness': (5, 80),
    'Insulin': (10, 600),
    'BMI': (15, 50)
}

for column, (min_val, max_val) in rules.items():
    # Remplacer les valeurs hors limites par la médiane
    data_split[column] = np.where(
        (data_split[column] < min_val) | (data_split[column] > max_val),
        data_split[column].median(),
        data_split[column]
    )

# 2.2 : Isolation Forest pour détecter les outliers multidimensionnels
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']
X = data_split[features]
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(X)
data_split['Outlier'] = outlier_labels

# Supprimer les outliers (ou les conserver si votre modèle est robuste)
data_cleaned = data_split[data_split['Outlier'] != -1].copy()
data_cleaned = data_cleaned.drop(columns=['Outlier'])

# Étape 3 : Transformation des variables (réduire l'asymétrie)
# Appliquer une transformation logarithmique à Insulin et Pedigree (qui sont asymétriques)
#data_cleaned['Insulin'] = np.log1p(data_cleaned['Insulin'])  # log1p pour éviter log(0)
data_cleaned['Pedigree'] = np.log1p(data_cleaned['Pedigree'])

# Étape 4 : Standardisation des données
scaler = StandardScaler()
data_cleaned[features] = scaler.fit_transform(data_cleaned[features])

# Étape 5 : Vérifier l'équilibre des classes
print("Répartition des classes avant équilibrage :")
print(data_cleaned['Outcome'].value_counts())

# Équilibrer les classes avec SMOTE (si nécessaire)
X = data_cleaned[features]
y = data_cleaned['Outcome']
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Créer un nouveau DataFrame avec les données équilibrées
data_balanced = pd.DataFrame(X_balanced, columns=features)
data_balanced['Outcome'] = y_balanced

print("Répartition des classes après équilibrage :")
print(data_balanced['Outcome'].value_counts())

# Étape 6 : Sélection des features (facultatif)
# Utiliser la matrice de corrélation pour identifier les variables redondantes
correlation_matrix = data_balanced.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Matrice de corrélation après pré-traitement")
plt.show()

# Par exemple, si SkinThickness et Insulin sont fortement corrélés (> 0.7), on peut supprimer une des deux
# Ici, pas de corrélation très forte, donc on garde toutes les variables

# Sauvegarder les données pré-traitées
data_balanced.to_csv('C:\\Users\\USER\\Desktop\\projet data mining\\diabetes_preprocessed.csv', index=False)
print("Données pré-traitées sauvegardées dans 'diabetes_preprocessed.csv'")