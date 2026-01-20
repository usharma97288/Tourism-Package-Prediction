from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "usharma97288/Tourism-Package-Prediction-Prj"
repo_type = "dataset"
file_to_upload = "Tourism-Package-Prediction/data/tourism.csv"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Step 2: Upload the specific file
# Use upload_file for explicit single file upload, which is more direct than upload_folder
# when only one specific file is being added/updated.
try:
    api.upload_file(
        path_or_fileobj=file_to_upload,
        path_in_repo=os.path.basename(file_to_upload), # Upload with its original filename
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Upload {os.path.basename(file_to_upload)}"
    )
    print(f"Successfully uploaded {os.path.basename(file_to_upload)} to {repo_id}.")
except HfHubHTTPError as e:
    if "no files found to make a commit" in str(e):
        print(f"No changes detected for {os.path.basename(file_to_upload)}. Skipping upload.")
    else:
        print(f"Error uploading file: {e}")
except Exception as e:
    print(f"An unexpected error occurred during file upload: {e}")
