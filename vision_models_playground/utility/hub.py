import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, metadata_save

from vision_models_playground.utility.load_models import load_best_model


def push_model_to_hub(
        repo_id: str,
        model_path: str,
):
    # Read the config from the model_path
    config_path = Path(model_path) / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Get the repo name
    _, repo_name = repo_id.split("/")

    # Create the api object
    api = HfApi()

    # Create the repo
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
    )

    # Download files
    repo_local_path = Path(snapshot_download(repo_id=repo_id))

    # Copy the files from model_path to repo_local_path
    shutil.copytree(model_path, repo_local_path / "model", dirs_exist_ok=True)

    metadata = {}
    metadata["tags"] = ["computer_vision", "vision_models_playground", "custom-implementation"]

    model_card_path = 'resources/model_card_template.txt'
    with open(model_card_path, "r") as f:
        model_card = f.read()
    model_card = model_card.replace("<config['class_name']>", config['class_name'])
    model_card = model_card.replace("<config['module_name']>", config['module_name'])

    # Create a readme.md with the model path
    readme_path = repo_local_path / "README.md"

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(model_card)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

    # Push everything to the hub
    api.upload_folder(
        repo_id=repo_id,
        folder_path=repo_local_path,
        path_in_repo=".",
    )


def load_vmp_model_from_hub(repo_id):
    # Download files
    repo_local_path = Path(snapshot_download(repo_id=repo_id))

    # Load best model
    model_path = repo_local_path / "model"
    model = load_best_model(str(model_path))

    return model

