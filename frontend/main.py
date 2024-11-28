import streamlit as st
import pandas as pd
import requests

# Configuration de l'interface
st.title("Prédiction de Churn")
st.write("Téléchargez un fichier CSV ou entrez des informations sur un client.")

# Chargement du fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    # Lire le fichier CSV
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données chargées :")
    st.dataframe(data.head())

    # Envoi des données au backend
    if st.button("Prédire"):
        response = requests.post("http://127.0.0.1:8080/predict/csv", files={"file": uploaded_file})

        if response.status_code == 200:
            predictions = response.json()["predictions"]
            st.write("Résultats des prédictions :")
            st.write(predictions)
        else:
            st.write("Erreur dans la requête au backend.")

