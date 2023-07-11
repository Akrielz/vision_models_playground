import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from huggingface_hub import HfApi, snapshot_download, metadata_save

from vision_models_playground.utility.load_models import load_best_model


def _create_readme(
        readme_path: Path,
        config: Dict[str, Any],
        report_path: Optional[Path] = None
):
    # If the report path was given, read the report
    metrics_log = ""
    if report_path is not None:
        with open(report_path, "r") as f:
            metrics_log = f.read()

    metadata = {}
    metadata["tags"] = ["computer_vision", "vision_models_playground", "custom-implementation"]

    model_card_path = 'resources/model_card_template.txt'
    with open(model_card_path, "r") as f:
        model_card = f.read()

    # Find all the <variable> in using a regex
    variable_regex = r"<(.*?)>"
    matches = re.findall(variable_regex, model_card)

    for variables in matches:
        model_card = model_card.replace(f"<{variables}>", eval(variables))

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(model_card)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)


def push_model_to_hub(
        repo_id: str,
        model_path: str,
        eval_path: Optional[str] = None,
        rewrite_readme: bool = False,
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
    api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
    )

    # Download files
    repo_local_path = Path(snapshot_download(repo_id=repo_id))

    # Copy the files from model_path to repo_local_path
    shutil.copytree(model_path, repo_local_path / "model", dirs_exist_ok=True)

    # If the eval path was given, copy the files from eval_path to repo_local_path
    if eval_path is not None:
        shutil.copytree(eval_path, repo_local_path / "eval", dirs_exist_ok=True)

    # Check if readme already exists
    readme_path = repo_local_path / "README.md"

    if not readme_path.exists() or rewrite_readme:
        report_path = repo_local_path / "eval" / "report.md" if eval_path is not None else None
        _create_readme(readme_path, config, report_path)

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


if __name__ == "__main__":
    push_model_to_hub(
        repo_id="Akriel/ResNetYoloV1",
        model_path="models/train/ResNetYoloV1/2023-07-06_14-37-23",
        eval_path="models/eval/ResNetYoloV1/2023-07-08_17-01-11",
        rewrite_readme=True,
    )
