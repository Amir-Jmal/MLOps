import streamlit as st
import pandas as pd
import requests
from joblib import load
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
#from backend.src.clean_data_csv import clean_data_csv
# def clean_data_csv(df,cols_to_drop):
#     df_cleaned = df.drop(columns=cols_to_drop)
#     numerical_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
#     scaler = MinMaxScaler()
#     df_cleaned[numerical_columns] = scaler.fit_transform(df_cleaned[numerical_columns])
#     yes_no_columns = [col for col in df_cleaned.columns if df_cleaned[col].isin(['Yes', 'No']).all()]
#     label_encoder = LabelEncoder()
#     for col in yes_no_columns:
#         df_cleaned[col] = label_encoder.fit_transform(df_cleaned[col])
#     categorical_columns = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
#     encoded_df = pd.get_dummies(df_cleaned[categorical_columns], drop_first=True)
#     df_cleaned = df_cleaned.drop(columns=categorical_columns).join(encoded_df)
#     bool_columns = df_cleaned.select_dtypes(include='bool').columns
#     df_cleaned[bool_columns] = df_cleaned[bool_columns].astype(int)
#     return df_cleaned

# Configuration de l'interface
st.title("Prédiction de Churn")
st.write("Téléchargez un fichier CSV ou entrez des informations sur un client.")

# Chargement du fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
API_URL = "http://127.0.0.1:8000/predict/csv"

if uploaded_file is not None:
    # Lire le fichier CSV
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données chargées :")
    st.dataframe(data.head())

    # Envoi des données au backend
    if st.button("Prédire"):
        # Envoyer les données au serveur FastAPI
        with st.spinner("En cours de traitement..."):
            file = {"file": uploaded_file.getvalue()}
            res = requests.post("http://localhost:8080/predict/csv", files=file, verify=False)
            predictions = res.json().get("predictions")


            # else:
            #     model_loaded = load("C:/Users/User/PycharmProjects/Mlops-Churn Prediction/weights/random_forest_model.joblib")
            #     cols_to_drop = ['Customer ID', 'Gender', 'Age', 'Zip Code', 'Latitude', 'Longitude', 'City',
            #                     'Churn Category',
            #                     'Churn Score', 'Churn Reason', 'Customer Status', 'Quarter', 'State', 'Country']
            #     clean_df = clean_data_csv(data, cols_to_drop)
            #     selected_features = ['Avg Monthly GB Download',
            #          'CLTV',
            #          'Contract_Two Year',
            #          'Monthly Charge',
            #          'Number of Referrals',
            #          'Population',
            #          'Satisfaction Score',
            #          'Tenure in Months',
            #          'Total Charges',
            #          'Total Long Distance Charges',
            #          'Total Revenue',
            #          'Internet Type_Fiber Optic']
            #     preprocessed_data = clean_df[selected_features]
            #     predictions = model_loaded.predict(preprocessed_data)

            st.write("Résultats des prédictions :")

            # Calcul des pourcentages des classes
            prediction_counts = pd.Series(predictions).value_counts(normalize=True) * 100
            prediction_counts.index = ['Not Churn', 'Churn']  # Remplacer les index numériques par des labels
            st.write("Résultats des prédictions (en pourcentage) :")
            st.write(prediction_counts)

            fig, ax = plt.subplots()
            prediction_counts.plot(kind='bar', ax=ax, color=['skyblue', 'orange'], alpha=0.7)
            ax.set_title("Pourcentage des classes prédites")
            ax.set_xlabel("Classe")
            ax.set_ylabel("Pourcentage")
            ax.set_xticklabels(prediction_counts.index, rotation=0)
            # Affichage dans Streamlit
            st.pyplot(fig)

            # Filtrer les clients ayant churn
            data['Prediction'] = predictions
            churn_data = data[data['Prediction'] == 1]
            st.write("Données pour la classe Churn :")
            st.dataframe(churn_data)