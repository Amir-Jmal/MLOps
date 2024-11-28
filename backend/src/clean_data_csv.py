import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np


def clean_data_csv(df,cols_to_drop):
    df_cleaned = df.drop(columns=cols_to_drop)
    numerical_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
    scaler = MinMaxScaler()
    df_cleaned[numerical_columns] = scaler.fit_transform(df_cleaned[numerical_columns])
    yes_no_columns = [col for col in df_cleaned.columns if df_cleaned[col].isin(['Yes', 'No']).all()]
    label_encoder = LabelEncoder()
    for col in yes_no_columns:
        df_cleaned[col] = label_encoder.fit_transform(df_cleaned[col])
    categorical_columns = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
    encoded_df = pd.get_dummies(df_cleaned[categorical_columns], drop_first=True)
    df_cleaned = df_cleaned.drop(columns=categorical_columns).join(encoded_df)
    bool_columns = df_cleaned.select_dtypes(include='bool').columns
    df_cleaned[bool_columns] = df_cleaned[bool_columns].astype(int)
    return df_cleaned