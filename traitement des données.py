import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# Étape 1 : Charger les données
data = pd.read_csv('C:\\Users\\USER\\Desktop\\projet data minig\\diabetes.csv')
if data.shape[1] == 1:
    data_split = data[data.columns[0]].str.split(';', expand=True)
    data_split.columns = ['Pregnanciess', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age', 'Outcome']
else:
    data_split = data

data_split = data_split.astype(float)

# Étape 2 : Gestion des zéros impossibles (valeurs manquantes déguisées)
# Remplacer les zéros par NaN pour les variables 
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
for column in columns_with_zeros:
    data_split[column] = data_split[column].replace(0, np.nan)

# Utiliser KNN Imputation pour toutes les colonnes,
imputer = KNNImputer(n_neighbors=5)
data_imputed = pd.DataFrame(imputer.fit_transform(data_split), columns=data_split.columns)

# Étape 3 : Gestion des valeurs aberrantes (règles médicales)
rules = {
    'Glucose': (40, 300),
    'BloodPressure': (40, 150),
    'SkinThickness': (5, 80),
    'Insulin': (10, 600),
    'BMI': (15, 50)
}

for column, (min_val, max_val) in rules.items():
    # Remplacer les valeurs hors limites par la médiane
    data_imputed[column] = np.where(
        (data_imputed[column] < min_val) | (data_imputed[column] > max_val),
        data_imputed[column].median(),
        data_imputed[column]
    )

# Étape 4 : Transformation des variables (tester avec et sans log pour Insulin)
# On va créer deux versions des données : une avec log sur Insulin, une sans

data_without_log = data_imputed.copy()

# Version sans transformation logarithmique sur Insulin (mais on garde pour Pedigree)
data_without_log['Pedigree'] = np.log1p(data_without_log['Pedigree'])

# Étape 5 : Standardisation des données
scaler = StandardScaler()
features = ['Pregnanciess', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']


data_without_log[features] = scaler.fit_transform(data_without_log[features])

# Étape 6 : Équilibrage des classes avec SMOTE

smote = SMOTE(random_state=42)

X_without_log = data_without_log[features]
y_without_log = data_without_log['Outcome']
X_without_log_balanced, y_without_log_balanced = smote.fit_resample(X_without_log, y_without_log)

data_without_log_balanced = pd.DataFrame(X_without_log_balanced, columns=features)
data_without_log_balanced['Outcome'] = y_without_log_balanced

selected_features = ['Pregnanciess', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Pedigree', 'Age']
data_final = data_without_log_balanced[selected_features + ['Outcome']]
# Étape 8 : Sauvegarder les données pré-traitées
data_final.to_csv("C:\\Users\\USER\\Desktop\\projet data minig\\diabetes_apres1traitement.csv", index=False)
print("Données pré-traitées sauvegardées dans 'diabetes_preprocessed.csv'")