from typing import Optional

import torch
from torch import nn

from vision_models_playground.utility.config import build_object_from_config_path


def load_model_from_weights(
        model: nn.Module,
        weights_path: str,
        device: Optional[torch.device] = None
):
    """
    Load the weights of a model

    Arguments
    ---------
    model: nn.Module
        The model to load the weights into. It needs to be the same model as the one used to save the weights

    weights_path: str
        The path to the weights

    device: Optional[torch.device]
        The device to load the weights to. If None, it will be loaded to the device the model is on
    """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )

    return model


def load_model_from_config(
        config_path: str,
        weights_path: str,
        device: Optional[torch.device] = None,
):
    """
    Load a model from a config file and weights

    Arguments
    ---------
    config_path: str
        The path to the config file

    weights_path: str
        The path to the weights

    device: Optional[torch.device]
        The device to load the weights to. If None, it will be loaded to the device the model is on
    """

    model = build_object_from_config_path(config_path)
    model = load_model_from_weights(model, weights_path, device)

    return model


def load_model_from_dir(
        save_dir: str,
        device: Optional[torch.device] = None,
        file_name: str = 'best'
):
    """
    Load the best model from a save directory

    Arguments
    ---------
    save_dir: str
        The directory to load the model from

    device: Optional[torch.device]
        The device to load the weights to. If None, it will be loaded to the device the model is on

    file_name: str
        The name of the file to load the weights from
    """
    config_path = f'{save_dir}/config.json'
    weights_path = f'{save_dir}/{file_name}.pt'

    return load_model_from_config(config_path, weights_path, device)