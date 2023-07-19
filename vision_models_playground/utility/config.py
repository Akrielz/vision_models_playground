import importlib
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

from vision_models_playground.pipelines.base import Pipeline


def config_wrapper(cls):
    def wrapper(*args, **kwargs):
        config = {'args': args, 'kwargs': kwargs, 'class_name': cls.__name__, 'module_name': cls.__module__}
        obj = cls(*args, **kwargs)
        obj._config = config

        return obj

    return wrapper


def build_object_from_config(config: Dict[str, Any]):
    _module = importlib.import_module(config['module_name'])
    _class = getattr(_module, config['class_name'])
    return _class(*config['args'], **config['kwargs'])


def build_object_from_config_path(config_path: str):
    config = json.load(open(config_path, 'r'))
    return build_object_from_config(config)


def get_config_from_object(obj: Any):
    config = getattr(obj, '_config', None)

    if config is None:
        raise ValueError('Object does not have a config attribute. Please use config_wrapper')

    return config


def config_to_json(config: Dict[str, Any], path: str):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)


def init_path(
        save_dir: Optional[str],
        object: Any,
        prefix: Optional[str] = None
):
    if save_dir is None:
        save_dir = '.'

    if prefix is None:
        prefix = ''

    current_date = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    save_dir = f'{save_dir}/{prefix}/{object.__class__.__name__}/{current_date}'
    save_dir = os.path.normpath(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = f'{save_dir}/config.json'
    path = os.path.normpath(path)

    return path


def object_to_json(obj: Any, save_dir: Optional[str] = None):
    path = init_path(save_dir, obj)
    config = get_config_from_object(obj)
    config_to_json(config, path)


def pipeline_to_json(pipeline: Pipeline, save_dir: Optional[str] = None):
    path = init_path(save_dir, pipeline, prefix='model/pipelines')
    config = get_config_from_object(pipeline)
    config['args'] = []
    config_to_json(config, path)