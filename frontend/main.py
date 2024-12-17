import streamlit as st
import pandas as pd
import requests
from joblib import load
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt

# Configuration de l'interface
st.title("Prédiction de Churn")
st.write("Téléchargez un fichier CSV ou entrez des informations sur un client.")

# Chargement du fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
API_URL = "http://127.0.0.1:8000/predict/csv"

if uploaded_file is not None:

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

            st.write("Résultats des prédictions :")

            # Calcul des pourcentages des classes
            prediction_counts = pd.Series(predictions).value_counts(normalize=True) * 100
            prediction_counts.index = ['Not Churn', 'Churn']
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