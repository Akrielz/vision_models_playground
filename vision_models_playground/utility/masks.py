import torch


def create_triangular_mask(
        batch_size: int,
        length: int,
        device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Return a triangular mask with shape [b, n, n] where:
    b = batch_size
    n = length
    """

    b = batch_size
    n = length
    mask = torch.ones([b, n, n], dtype=torch.bool, device=device)
    mask = torch.triu(mask, diagonal=1)
    mask = ~mask
    return mask
