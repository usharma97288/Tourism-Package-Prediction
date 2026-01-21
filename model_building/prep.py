# for data manipulation
import pandas as pd
import numpy as np
import joblib

# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
# for handling class imbalance
from imblearn.over_sampling import SMOTE
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, login, create_repo
from huggingface_hub.utils import HfHubHTTPError
import sys # Import sys to access command-line arguments

# Define constants for the dataset and output paths
repo_id = "usharma97288/Tourism-Package-Prediction-Prj"
DATASET_PATH = f"hf://datasets/{repo_id}/tourism.csv"
LOCAL_DATA_DIR = "Tourism-Package-Prediction/data"

# Ensure the local data directory exists
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# Get HF_TOKEN from command-line argument or environment variable
# Priority given to command-line argument if provided
hf_token = None
if len(sys.argv) > 1:
    hf_token = sys.argv[1]
elif os.getenv("HF_TOKEN"):
    hf_token = os.getenv("HF_TOKEN")

print(f"HF_TOKEN status: {'Found' if hf_token else 'Not Found'}")
if not hf_token:
    print("HF_TOKEN environment variable or command-line argument not set. Exiting.")
    sys.exit(1) # Use sys.exit to properly exit the script

# Ensure Hugging Face login
try:
    login(token=hf_token)
    print("Successfully logged into Hugging Face.")
except Exception as e:
    print(f"Hugging Face login failed: {e}")
    sys.exit(1) # Exit if login fails

api = HfApi(token=hf_token)

# Ensure the repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type="dataset")
    print(f"Hugging Face repository '{repo_id}' already exists.")
except HfHubHTTPError as e:
    if e.response.status_code == 404: # Check for 404 Not Found error
        print(f"Hugging Face repository '{repo_id}' not found. Creating it...")
        create_repo(repo_id=repo_id, repo_type="dataset", private=False)
        print(f"Hugging Face repository '{repo_id}' created.")
    else:
        print(f"Error checking/creating Hugging Face repository: {e}")
        sys.exit(1) # Exit if repository issues persist
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1) # Exit if any unexpected error occurs



# 1. Load the dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# 2. Data Cleaning
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)
print("Dropped 'Unnamed: 0' and 'CustomerID' columns.")

# Handle Gender inconsistency
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
print("Handled 'Gender' inconsistency.")

# 3. Feature Engineering
df['TotalVisitors'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
print("Created 'TotalVisitors' feature.")
df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, df['Age'].max()], labels=['18-30', '31-45', '46-60', '60+'], right=False)
print("Created 'AgeGroup' feature.")
df['IncomeGroup'] = pd.qcut(df['MonthlyIncome'], q=3, labels=['Low', 'Medium', 'High'])
print("Created 'IncomeGroup' feature.")

# 4. Outlier Handling
# Identify numerical columns for outlier capping (excluding 'Age' and target variable)
numerical_cols_for_outlier_analysis = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# Exclude 'ProdTaken' (target) and 'Age' (already inspected, no significant outliers or specific handling needed)
excluded_from_outlier_capping = ['ProdTaken', 'Age']

numerical_cols_for_outlier_analysis = [
    col for col in numerical_cols_for_outlier_analysis
    if col not in excluded_from_outlier_capping
]

for col in numerical_cols_for_outlier_analysis:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    print(f"Outliers capped for column: {col}")

# 5. Feature Encoding
# Convert boolean columns to int for consistent numerical scaling later
for col in df.select_dtypes(include='bool').columns:
    df[col] = df[col].astype(int)

nominal_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
print("One-hot encoded nominal categorical features.")

designation_order = ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP']
agegroup_order = ['18-30', '31-45', '46-60', '60+']
incomegroup_order = ['Low', 'Medium', 'High']

categories = [designation_order, agegroup_order, incomegroup_order]
ordinal_cols = ['Designation', 'AgeGroup', 'IncomeGroup']

encoder = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
df[ordinal_cols] = encoder.fit_transform(df[ordinal_cols])
print("Ordinal encoded categorical features: Designation, AgeGroup, IncomeGroup.")

# 6. Feature Scaling
# Identify numerical columns for scaling after encoding and feature engineering
all_numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
excluded_cols_from_scaling = ['ProdTaken'] # Only target should not be scaled
features_to_scale = [col for col in all_numerical_cols if col not in excluded_cols_from_scaling]

scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print("Numerical features scaled successfully.")

# 7. Data Splitting and SMOTE
target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Data split into training and testing sets.")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print("SMOTE applied to training data.")

# 8. Save and Upload Processed Files
output_files = {
    "X_train_new.csv": X_resampled,
    "X_test_new.csv": X_test,
    "y_train_new.csv": y_resampled,
    "y_test_new.csv": y_test
}

for filename, data in output_files.items():
    local_filepath = os.path.join(LOCAL_DATA_DIR, filename)
    if isinstance(data, pd.DataFrame):
        data.to_csv(local_filepath, index=False)
    else: # For Series (y_resampled, y_test)
        data.to_csv(local_filepath, index=False, header=True) # Ensure header is written for y files
    print(f"Saved {local_filepath} locally.")

    try:
        api.upload_file(
            path_or_fileobj=local_filepath,
            path_in_repo=filename, # Upload with its original filename to HF repo root
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload {filename} (processed data)"
        )
        print(f"Successfully uploaded '{filename}' to Hugging Face repository '{repo_id}'.")
    except Exception as e:
        print(f"Error uploading '{filename}' to Hugging Face: {e}")

FEATURE_COLUMNS = X_train.columns.tolist()
joblib.dump(FEATURE_COLUMNS, "feature_columns.joblib")

try:
    api.upload_file(
        path_or_fileobj="feature_columns.joblib",
        path_in_repo="feature_columns.joblib",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload feature schema"
    )
    print("Successfully uploaded feature_columns.joblib to Hugging Face.")
except Exception as e:
    print(f"Error uploading feature_columns.joblib: {e}")

print("All data processing and upload completed.")
