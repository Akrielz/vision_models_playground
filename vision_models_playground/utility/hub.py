import json
import re
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from huggingface_hub import HfApi, snapshot_download, metadata_save

import vision_models_playground.utility.config as utility_config
import vision_models_playground.utility.load_models as utility_load_models


def _create_readme(
        readme_path: Path,
        model_config: Dict[str, Any],
        repo_id: str,
        report_path: Optional[Path] = None,
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
        pipeline_path: Optional[str] = None,
        rewrite_readme: bool = False,
):
    # Read the config from the model_path
    config_path = Path(model_path) / "config.json"
    with open(config_path, "r") as f:
        model_config = json.load(f)

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

    if pipeline_path is not None:
        shutil.copytree(pipeline_path, repo_local_path / "pipeline", dirs_exist_ok=True)

    # Check if readme already exists
    readme_path = repo_local_path / "README.md"

    if not readme_path.exists() or rewrite_readme:
        report_path = repo_local_path / "eval" / "report.md" if eval_path is not None else None
        _create_readme(readme_path, model_config, repo_id, report_path)

    # Push everything to the hub
    api.upload_folder(
        repo_id=repo_id,
        folder_path=repo_local_path,
        path_in_repo=".",
    )


def load_vmp_model_from_hub(
        repo_id: str,
        file_name: str = "best",
):
    # Download files
    repo_local_path = Path(snapshot_download(repo_id=repo_id))

    # Load best model
    model_path = repo_local_path / "model"
    model = utility_load_models.load_model_from_dir(str(model_path), file_name=file_name)

    return model


def load_vmp_pipeline_from_hub(
        repo_id: str,
        file_name: str = "best",
):
    # Get model from hub
    model = load_vmp_model_from_hub(repo_id, file_name=file_name)

    # Get download path
    repo_local_path = Path(snapshot_download(repo_id=repo_id))

    # Get pipeline
    pipeline_config_path = repo_local_path / "pipeline" / "config.json"
    pipeline = utility_config.build_object_from_config_path(str(pipeline_config_path))
    pipeline.set_model(model)

    return pipeline


if __name__ == "__main__":
    # push_model_to_hub(
    #     repo_id="Akriel/ResNetYoloV1",
    #     model_path="models/train/ResNetYoloV1/2023-07-06_14-37-23",
    #     eval_path="models/eval/ResNetYoloV1/2023-07-08_17-01-11",
    #     pipeline_path="models/pipelines/YoloV1Pipeline/2023-07-13_20-08-34",
    #     rewrite_readme=True,
    # )

    pipeline = load_vmp_pipeline_from_hub("Akriel/ResNetYoloV1")
    print(pipeline)

