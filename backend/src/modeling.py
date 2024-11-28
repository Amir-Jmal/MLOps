from data_preprocessing_training import *
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


path=""
data = pd.read_csv(path)
cols_to_drop = ['Customer ID', 'Gender', 'Age','Zip Code', 'Latitude', 'Longitude','City', 'Churn Category','Churn Score','Churn Reason', 'Customer Status','Quarter','State','Country']
df=preprocessing(data,cols_to_drop)
X_train, X_test, y_train, y_test,selected_features = feature_selection_with_random_forest(df, 'Churn Label')
X_train, X_test, y_train, y_test=apply_smote(X_train, X_test, y_train, y_test)
X_test, y_test = test_balanced_sample(X_test, y_test)


os.environ['MLFLOW_TRACKING_USERNAME']= "Amir-Jmal"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "85842cb6984ed595d31c00d48f8d7aef1cafa278"
#setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/api/v1/repo-buckets/s3/Amir-Jmal') #your mlfow tracking uri
mlflow.set_experiment("churn-prediction-experiment")

mlflow.sklearn.autolog(disable=True)


# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    'XGB': XGBClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0)
}

# Dictionary to store scores
# scores = {
#     "Model": [],
#     "Train Accuracy": [],
#     "Test Accuracy": [],
#     "Train F1 Score": [],
#     "Test F1 Score": []
# }

# Set up MLflow experiment
#mlflow.set_experiment("ML_Model_Comparison")  # Set a custom experiment name

# Loop through models to train and evaluate
for model_name, model in models.items():
    print(f"Training {model_name}...")

    with mlflow.start_run():  # Start a new MLflow run for each model
        # Log model parameters
        mlflow.log_param("model", model_name)
        # train_features_name = f'{X_train=}'.split('=')[0]
        # train_label_name = f'{y_train=}'.split('=')[0]
        # mlflow.set_tag(key="train_features_name",value= train_features_name)
        # mlflow.set_tag(key= "train_label_name",value=train_label_name)
        mlflow.log_param("selected_features", ", ".join(selected_features))

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate scores
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')

        # Log metrics to MLflow
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("train_f1_score", train_f1)
        mlflow.log_metric("test_f1_score", test_f1)

        # Add the results to the dictionary for later display
        # scores["Model"].append(model_name)
        # scores["Train Accuracy"].append(train_accuracy)
        # scores["Test Accuracy"].append(test_accuracy)
        # scores["Train F1 Score"].append(train_f1)
        # scores["Test F1 Score"].append(test_f1)

        # Log the model (based on the model type)
        if model_name == 'XGB':
            mlflow.xgboost.log_model(model, "model")
        elif model_name == 'CatBoost':
            mlflow.catboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        # Compute and display confusion matrix
        # cm = confusion_matrix(y_test, y_test_pred)
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
        #             xticklabels=["Not Churn", "Churn"],  # Update with your classes
        #             yticklabels=["Not Churn", "Churn"])  # Update with your classes
        # plt.title(f'Confusion Matrix for {model_name}')
        # plt.xlabel('Predicted')
        # plt.ylabel('Actual')
        # plt.show()

        # Print classification report
        # print(f"Classification Report for {model_name}:\n")
        # print(classification_report(y_test, y_test_pred))
        # print("=" * 180)  # Separator for clarity

