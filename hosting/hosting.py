from huggingface_hub import HfApi
from pathlib import Path
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Resolve deployment folder relative to repo root
deployment_dir = Path(__file__).resolve().parent.parent / "deployment"

print(
    f"Uploading folder {deployment_dir} "
    f"to Hugging Face Space: usharma97288/Tourism-Package-Prediction-App"
)

api.upload_folder(
    folder_path=str(deployment_dir),
    repo_id="usharma97288/Tourism-Package-Prediction-App",
    repo_type="space",
    path_in_repo="",
    commit_message="Deploying Streamlit app for Tourism Package Prediction"
)

print("Deployment artifacts uploaded successfully to Hugging Face Space.")
