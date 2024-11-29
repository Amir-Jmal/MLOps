# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import mlflow
import mlflow.pyfunc
import os
from backend.src.clean_data_csv import clean_data_csv
import uvicorn

# Configuration de l'environnement MLflow
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/Amir-Jmal/MLOps-Churn-Prediction.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']= "Amir-Jmal"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "85842cb6984ed595d31c00d48f8d7aef1cafa278"

# Initialisation de l'application FastAPI

app = FastAPI()

# Ajouter CORS pour accepter les requêtes depuis n'importe quelle origine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Permet tous les headers
)



# Fonction pour charger le meilleur modèle depuis MLflow
def get_best_model():
    all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
    df_mlflow = mlflow.search_runs(experiment_ids=all_experiments, filter_string="metrics.test_f1_score <1")
    run_id = df_mlflow.loc[df_mlflow['metrics.test_f1_score'].idxmax()]['run_id']
    logged_model = f'runs:/{run_id}/model'
    model = mlflow.pyfunc.load_model(logged_model)
    selected_features = mlflow.get_run(run_id).data.params.get("selected_features", "").split(",")

    return model, selected_features


# Charger le modèle dès le démarrage de l'application
model,selected_feature = get_best_model()

# Endpoint de test pour vérifier si l'API fonctionne
@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API de prédiction de churn"}

# Endpoint pour les fichiers CSV : prédiction pour plusieurs transactions
@app.post("/predict/csv")
def predict_csv(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    cols_to_drop = ['Customer ID', 'Gender', 'Age', 'Zip Code', 'Latitude', 'Longitude', 'City', 'Churn Category',
                    'Churn Score', 'Churn Reason', 'Customer Status', 'Quarter', 'State', 'Country']
    clean_df = clean_data_csv(data,cols_to_drop)
    selected_features = [feature.strip() for feature in selected_feature]
    preprocessed_data = clean_df[selected_features]
    predictions = model.predict(preprocessed_data)
    return {"predictions": predictions.tolist()}

# Point d'entrée pour lancer l'API
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
