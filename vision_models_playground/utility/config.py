import importlib
import json
from typing import Dict, Any


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
