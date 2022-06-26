import torch
from torchvision.models import resnet34, resnet50

from utility.functions import get_number_of_parameters


def main():
    block = resnet50(pretrained=False, num_classes=10)
    num_params = get_number_of_parameters(block)
    print(f"Number of parameters: {num_params}")

    x = torch.randn(1, 3, 300, 300)
    out = block(x)
    print(out.shape)


if __name__ == "__main__":
    main()