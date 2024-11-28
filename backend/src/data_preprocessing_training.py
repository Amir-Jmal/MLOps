import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE


def preprocess_and_fill_missing(df, target_column, val_size=0.2):
    # Sélectionner les colonnes catégorielles (de type 'object' ou 'category') dans le DataFrame
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Retirer la colonne cible des colonnes catégorielles (car elle ne doit pas être encodée)
    categorical_columns.remove(target_column)
    # Appliquer la fonction pd.get_dummies pour encoder les colonnes catégorielles en variables indicatrices
    encoded_df = pd.get_dummies(df[categorical_columns], drop_first=True)
    # Remplacer les colonnes catégorielles d'origine par les colonnes encodées dans le DataFrame
    data = df.drop(columns=categorical_columns).join(encoded_df)
    # Créer un masque pour identifier les valeurs manquantes dans la colonne cible
    missing_mask = data[target_column].isnull()
    # Séparer les données en ensembles d'entraînement (sans valeurs manquantes) et de test (avec valeurs manquantes)
    X_train = data[~missing_mask].drop(target_column, axis=1)
    y_train = data[~missing_mask][target_column]
    X_test = data[missing_mask].drop(target_column, axis=1)
    # Diviser l'ensemble d'entraînement en sous-ensembles d'entraînement et de validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=val_size
    )
    #print(f"Distribution des classes {target_column}  train:\n{y_train_split.value_counts()}\n")
    #print(f"Distribution des classes  {target_column}  val:\n{y_val_split.value_counts()}\n\n")
    # Entraîner un modèle de classification (Random Forest)
    model = RandomForestClassifier()
    model.fit(X_train_split, y_train_split)
    # Prédire les valeurs manquantes dans la colonne cible
    predictions = model.predict(X_test)
    # Remplacer les valeurs manquantes par les prédictions obtenues
    data.loc[missing_mask, target_column] = predictions
    # Calculer l'exactitude du modèle sur l'ensemble de validation
    accuracy_val = accuracy_score(y_val_split, model.predict(X_val_split))
    #print(f"Validation Accuracy of the model {target_column}: {accuracy_val:.2f}")
    return data

def preprocessing(data, cols_to_drop):
    # Supprimer les colonnes inutiles ou non pertinentes
    df_cleaned = data.drop(columns=cols_to_drop)
    # Sélectionner les colonnes numériques (de type 'float64' et 'int64')
    numerical_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Normaliser les colonnes numériques en utilisant StandardScaler
    # scaler = StandardScaler()
    # df_cleaned[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    # Normaliser les colonnes numériques en utilisant MinMaxScaler pour les ramener entre 0 et 1
    scaler = MinMaxScaler()
    df_cleaned[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    # Identifier les colonnes contenant uniquement les valeurs 'Yes' et 'No'
    yes_no_columns = [col for col in df_cleaned.columns if df_cleaned[col].isin(['Yes', 'No']).all()]
    # Initialiser un label encoder pour convertir les colonnes 'Yes' et 'No' en 0 et 1
    label_encoder = LabelEncoder()
    for col in yes_no_columns:
        df_cleaned[col] = label_encoder.fit_transform(df_cleaned[col])
    # Identifier les colonnes catégorielles restantes
    categorical_columns = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
    # Obtenir une liste des colonnes avec des valeurs manquantes
    columns_with_missing_values = df_cleaned.columns[df_cleaned.isnull().any()]
    # Initialiser une liste pour stocker les DataFrames après traitement des valeurs manquantes
    processed_dfs = []
    # Parcourir chaque colonne ayant des valeurs manquantes
    for col in columns_with_missing_values:
        # Créer un DataFrame temporaire avec uniquement la colonne à traiter
        temp_df = df_cleaned.drop([c for c in columns_with_missing_values if c != col ], axis=1)
        # Appliquer la fonction preprocess_and_fill_missing pour combler les valeurs manquantes
        preprocessed_df = preprocess_and_fill_missing(temp_df, col)
        # Ajouter le DataFrame traité à la liste
        processed_dfs.append(preprocessed_df)

    # Combiner tous les DataFrames traités en un seul
    final_df = processed_dfs[0]
    for df in processed_dfs[1:]:
        final_df = final_df.combine_first(df)
    # Supprimer les doublons s'il y en a
    if final_df.duplicated().sum() > 0:
        final_df = final_df.drop_duplicates()
    # Rechercher à nouveau les colonnes catégorielles après traitement
    categorical_columns = final_df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Encoder les colonnes catégorielles avec get_dummies
    encoded_df = pd.get_dummies(final_df[categorical_columns], drop_first=False)
    # Remplacer les colonnes catégorielles d'origine par les colonnes encodées
    df = final_df.drop(columns=categorical_columns).join(encoded_df)
    # Identifier les colonnes de type booléen et les convertir en entiers (0 ou 1)
    bool_columns = df.select_dtypes(include='bool').columns
    df[bool_columns] = df[bool_columns].astype(int)
    return df
def feature_selection_with_random_forest(df, target_column):
    # Split the data into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Train a Random Forest model to assess feature importance
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    # Use SelectFromModel to select features based on importance
    selector = SelectFromModel(rf, threshold="mean", prefit=True)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    # Print selected features
    selected_features = X.columns[selector.get_support()]
    #print(f"Selected features: {list(selected_features)}")
    return X_train_selected, X_test_selected, y_train, y_test,list(selected_features)

def apply_smote(X_train, X_test, y_train, y_test):
    # Initialiser SMOTE
    smote = SMOTE()
    # Appliquer SMOTE pour générer des exemples synthétiques pour la classe minoritaire
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    # print(f"Distribution des classes avant SMOTE :\n{y_train.value_counts()} \n")
    # print(f"Distribution des classes après SMOTE :\n{pd.Series(y_train_resampled).value_counts()}\n")
    return X_train_resampled, X_test, y_train_resampled, y_test

def test_balanced_sample(X_test, y_test):
    # Appliquer le sous-échantillonnage pour équilibrer l'échantillon de test
    sampler = RandomUnderSampler()
    churn_label_counts = y_test.value_counts()
    print("Original class distribution:")
    print(churn_label_counts)

    # Appliquer la méthode d'équilibrage sur les données de test
    X_test_balanced, y_test_balanced = sampler.fit_resample(X_test, y_test)
    churn_label_counts = y_test_balanced.value_counts()
    print("Balanced class distribution:")
    print(churn_label_counts)

    return X_test_balanced, y_test_balanced