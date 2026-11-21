from huggingface_hub import HfApi

api = HfApi()

repo_id = "Dinox16/phobert-sentiment"

api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    private=False,
    exist_ok=True
)

api.upload_folder(
    folder_path="D:/Cousers/CS 467/DA-CS467/result/DTU-CS/HGK/best_model",
    repo_id=repo_id,
    repo_type="model"
)