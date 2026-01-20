import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from huggingface_hub import HfApi, hf_hub_download, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow
import sys

# --- Configuration ---
# Hugging Face repository details for data and model storage
HF_REPO_ID = "usharma97288/Tourism-Package-Prediction-Prj"
MODEL_FILENAME = "tourism_xgb_model_grid_search.joblib"
MODEL_PATH_IN_REPO = f"models/{MODEL_FILENAME}"

# Initialize HfApi (HF_TOKEN is assumed to be set as an environment variable)
api = HfApi(token=os.getenv("HF_TOKEN"))


# --- Load Data from Hugging Face ---
try:
    print(f"Downloading X_train_new.csv from {HF_REPO_ID}...")
    X_train = pd.read_csv(hf_hub_download(repo_id=HF_REPO_ID, filename="X_train_new.csv", repo_type="dataset"))

    print(f"Downloading y_train_new.csv from {HF_REPO_ID}...")
    # .squeeze() to convert DataFrame to Series for consistent target variable handling
    y_train = pd.read_csv(hf_hub_download(repo_id=HF_REPO_ID, filename="y_train_new.csv", repo_type="dataset")).squeeze()

    print(f"Downloading X_test_new.csv from {HF_REPO_ID}...")
    X_test = pd.read_csv(hf_hub_download(repo_id=HF_REPO_ID, filename="X_test_new.csv", repo_type="dataset"))

    print(f"Downloading y_test_new.csv from {HF_REPO_ID}...")
    # .squeeze() to convert DataFrame to Series for consistent target variable handling
    y_test = pd.read_csv(hf_hub_download(repo_id=HF_REPO_ID, filename="y_test_new.csv", repo_type="dataset")).squeeze()

    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data from Hugging Face: {e}")
    sys.exit(1)

# The data (X_train, y_train, X_test, y_test) is already preprocessed (scaled, encoded, SMOTEd)
# by the `prep.py` script. Therefore, we can directly use it for model training without further
# preprocessing steps within this script's pipeline.

# Define base XGBoost model
# enable_categorical=True is important for XGBoost >= 1.6 with pre-encoded categorical features
# Since y_train was balanced using SMOTE, scale_pos_weight is not explicitly set here.
xgb_model = xgb.XGBClassifier(random_state=42, enable_categorical=True, eval_metric='logloss')

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
}

# Start MLflow run to track hyperparameter tuning and model evaluation
with mlflow.start_run():
    print("Starting GridSearchCV for hyperparameter tuning...")
    # Use 'roc_auc' as the scoring metric given the potential class imbalance (even after SMOTE)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3, n_jobs=-1, scoring='roc_auc', verbose=0)
    grid_search.fit(X_train, y_train)
    print("GridSearchCV completed.")

    # Log best parameters and score to the main MLflow run
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_roc_auc_score", grid_search.best_score_)

    # Store the best model found by GridSearchCV
    best_model = grid_search.best_estimator_

    # Make predictions and evaluate
    y_pred_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_pred_train = (y_pred_train_proba >= 0.5).astype(int) # Using 0.5 as a default classification threshold

    y_pred_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_pred_test_proba >= 0.5).astype(int) # Using 0.5 as a default classification threshold

    # Generate classification reports
    train_report = classification_report(y_train, y_pred_train, output_dict=True)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)

    # Calculate ROC-AUC scores
    train_roc_auc = roc_auc_score(y_train, y_pred_train_proba)
    test_roc_auc = roc_auc_score(y_test, y_pred_test_proba)

    # Log comprehensive metrics to MLflow
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision_1": train_report['1']['precision'],
        "train_recall_1": train_report['1']['recall'],
        "train_f1-score_1": train_report['1']['f1-score'],
        "train_roc_auc": train_roc_auc,
        "test_accuracy": test_report['accuracy'],
        "test_precision_1": test_report['1']['precision'],
        "test_recall_1": test_report['1']['recall'],
        "test_f1-score_1": test_report['1']['f1-score'],
        "test_roc_auc": test_roc_auc
    })
    print("Metrics logged to MLflow.")

    # Save the best model locally
    joblib.dump(best_model, MODEL_FILENAME)
    print(f"Best model saved locally as '{MODEL_FILENAME}'.")

    # Log the model artifact with MLflow
    mlflow.log_artifact(MODEL_FILENAME, artifact_path="model")
    print(f"Model logged as MLflow artifact.")

    # Upload the best model to Hugging Face
    try:
        # Ensure the target 'models' folder exists in the Hugging Face repository
        # This check is more robust for CI/CD environments
        try:
            api.list_repo_tree(repo_id=HF_REPO_ID, repo_type="dataset", path="models")
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                print(f"'models/' directory not found in {HF_REPO_ID}. Creating it...")
                api.create_commit(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    operations=[], # No actual file content, just creating a directory implicitly
                    commit_message="Create models directory"
                )
            else:
                raise e # Re-raise if it's another kind of error

        api.upload_file(
            path_or_fileobj=MODEL_FILENAME,
            path_in_repo=MODEL_PATH_IN_REPO, # Save in a 'models' subfolder within the repo
            repo_id=HF_REPO_ID,
            repo_type="dataset", # Using 'dataset' repo type as consistent with existing notebook structure
            commit_message=f"Upload best trained XGBoost model from MLflow run {mlflow.active_run().info.run_id}"
        )
        print(f"Best model uploaded to Hugging Face repository '{HF_REPO_ID}' at '{MODEL_PATH_IN_REPO}'.")
    except Exception as e:
        print(f"Error uploading model to Hugging Face: {e}")

print("Model training, hyperparameter tuning, and registration process completed.")
