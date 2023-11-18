from typing import List, Any, Literal

import torch
from torch import Tensor


def anything_to_device(inputs: Any, device: torch.device):
    """
    Recursively move a dictionary / list of tensors to a device.

    Arguments:
        inputs (dict or list or tensor): Dictionary / list of tensors
        device (torch.device): Device to move the tensors to

    Returns:
        dict or list: Dictionary / list of tensors on the device
    """

    if isinstance(inputs, Tensor):
        return inputs.to(device)

    if hasattr(inputs, 'to'):
        try:
            return inputs.to(device)
        except Exception:
            pass

        return inputs

    if isinstance(inputs, dict):
        inputs = {k: anything_to_device(v, device) for k, v in inputs.items()}
        return inputs

    if isinstance(inputs, list):
        inputs = [anything_to_device(v, device) for v in inputs]
        return inputs

    if isinstance(inputs, tuple):
        inputs = tuple(anything_to_device(v, device) for v in inputs)
        return inputs

    if isinstance(inputs, set):
        inputs = {anything_to_device(v, device) for v in inputs}
        return inputs

    return inputs


def select_sub_batch(inputs: Any, idx):
    """
    Considering that we have a complex input with multiple tensors, we want to
    extract a sub-batch from it.

    This function is used to extract the sub-batch in the same complex structure
    as the input.

    This function assumes every leaf is a tensor and that all the tensors have
    the same batch size.
    """

    if isinstance(inputs, Tensor):
        return inputs[idx]

    if isinstance(inputs, dict):
        inputs = {k: select_sub_batch(v, idx) for k, v in inputs.items()}
        return inputs

    if isinstance(inputs, list):
        inputs = [select_sub_batch(v, idx) for v in inputs]
        return inputs

    if isinstance(inputs, tuple):
        inputs = tuple(select_sub_batch(v, idx) for v in inputs)
        return inputs

    if isinstance(inputs, set):
        inputs = {select_sub_batch(v, idx) for v in inputs}
        return inputs

    raise ValueError(f"Unknown type {type(inputs)}")


def infer_batch_size(inputs: Any):
    """
    Infer the batch size of the input.

    This function assumes every leaf is a tensor and that all the tensors have
    the same batch size.
    """

    if isinstance(inputs, Tensor):
        return inputs.shape[0]

    if isinstance(inputs, dict):
        return infer_batch_size(next(iter(inputs.values())))

    if isinstance(inputs, list):
        return infer_batch_size(inputs[0])

    if isinstance(inputs, tuple):
        return infer_batch_size(inputs[0])

    if isinstance(inputs, set):
        return infer_batch_size(next(iter(inputs)))

    raise ValueError(f"Unknown type {type(inputs)}")


def combine_to_batch(inputs: List[Any], mode: Literal['concat', 'stack'] = 'stack'):
    """
    Combine a list of complex structures to a single batch.

    This function assumes every leaf is a tensor and that all the tensors have
    the same batch size.
    """

    # Make sure all the lists have the same type of elements
    assert len(set([type(i) for i in inputs])) == 1, "All the inputs must have the same type"

    if isinstance(inputs[0], Tensor):
        if mode == 'concat':
            return torch.cat(inputs, dim=0)
        elif mode == 'stack':
            return torch.stack(inputs, dim=0)

    if isinstance(inputs[0], dict):
        return {k: combine_to_batch([v[k] for v in inputs], mode=mode) for k in inputs[0].keys()}

    if isinstance(inputs[0], list):
        return [combine_to_batch([v[i] for v in inputs], mode=mode) for i in range(len(inputs[0]))]

    if isinstance(inputs[0], tuple):
        return tuple(combine_to_batch([v[i] for v in inputs], mode=mode) for i in range(len(inputs[0])))

    if isinstance(inputs[0], set):
        return {combine_to_batch([v[i] for v in inputs], mode=mode) for i in range(len(inputs[0]))}

    raise ValueError(f"Unknown type {type(inputs[0])}")
