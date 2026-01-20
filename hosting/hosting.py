
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
print(f"Uploading folder Tourism-Package-Prediction/deployment to Hugging Face Space: usharma97288/Tourism-Package-Prediction-App")
api.upload_folder(
    folder_path="Tourism-Package-Prediction/deployment", # the local folder containing your files
    repo_id="usharma97288/Tourism-Package-Prediction-App",                      # the target repo
    repo_type="space",                                # dataset, model, or space
    path_in_repo="",                                  # optional: subfolder path inside the repo
    commit_message="Deploying Streamlit app for Tourism Package Prediction"
)
print("Deployment artifacts uploaded successfully to Hugging Face Space.")
