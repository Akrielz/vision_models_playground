# Vision Models - Playground

## Table of Contents

- [Description](#description)
- [ResNet](#resnet)
- [VisionTransformer](#visiontransformer)
- [Generative Adversarial Networks](#generative-adversarial-networks-gan)

## Description

This playground is a collection of vision models implemented by me from scratch
in PyTorch, with the purpose of getting a better understanding of the specific
papers and techniques used.

## ResNet

A classifier based on the [ResNet](https://arxiv.org/abs/1512.03385) architecture.

### Usage

Models can be initialized with pre-build or custom versions.

Pre-build models:

- ResNet18
- ResNet34
- ResNet50
- ResNet101
- ResNet152

Code example to initialize and use prebuild ResNet34

```python
import torch
from models.classifiers.resnet import build_resnet34

model = build_resnet34(num_classes=10, in_channels=3)

img = torch.randn(1, 3, 300, 300)  # <batch_size, in_channels, height, width>
preds = model(img)  # (1, 10) <batch_size, num_classes>
```

Code example to initialize ResNet34 using the custom ResNet

```python
import torch
from models.classifiers.resnet import ResNet, ResidualBlock

model = ResNet(
    in_channels=3,
    num_classes=10,
    num_layers=[3, 4, 6, 3],
    num_channels=[64, 128, 256, 512],
    block=ResidualBlock
)

img = torch.randn(1, 3, 300, 300)  # <batch_size, in_channels, height, width>
preds = model(img)  # (1, 10) <batch_size, num_classes>
```


### Parameters

- `in_channels`: int.  
The number of channels in the input image.


- `num_classes`: int.  
The number of predicted classes


- `num_layers`: List[int]  
The number of block layers in each stage.


- `num_channels`: List[int]  
The number of channels in each stage.  
Each stage will start with a stride of 2, connecting the previous stage channels 
with the current stage channels.


- `block`: Union[ResidualBlock, BottleneckBlock]  
The block type used.  
There are two pre-implemented block types: ResidualBlock and BottleneckBlock.  
Can be replaced with any custom block that has the following params in the constructor:
`in_channels`, `out_channels`, `stride`.

## Vision Transformer

A classifier based on the [Vision Transformer](https://openreview.net/pdf?id=YicbFdNTTy) 
architecture.

### Usage

Code example to initialize and use Vision Transformer

```python
import torch
from models.classifiers.vision_transformer import VisionTransformer

model = VisionTransformer(
    image_size=256,
    patch_size=32,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1,
    apply_rotary_emb=True,
)

img = torch.randn(1, 3, 256, 256)
preds = model(img)  # (1, 1000)
```

### Parameters

- `image_size`: int.  
Image size. If you have rectangular images, make sure your image size is the maximum of the width and height


- `patch_size`: int.  
Number of patches. `image_size` must be divisible by `patch_size`.  
The number of patches is: ` n = (image_size // patch_size) ** 2` and `n` **must be greater than 16**.


- `num_classes`: int.  
Number of classes to classify.


- `dim`: int.  
Last dimension of output tensor after linear transformation `nn.Linear(..., dim)`.


- `depth`: int.  
Number of Transformer blocks.


- `heads`: int.  
Number of heads in Multi-head Attention layer. 


- `mlp_dim`: int.  
Dimension of the MLP (FeedForward) layer. 


- `channels`: int, default `3`.  
Number of image's channels. 


- `dropout`: float between `[0, 1]`, default `0`.  
Dropout rate. 


- `emb_dropout`: float between `[0, 1]`, default `0`.   
Embedding dropout rate.


- `dim_head`: int, default to `64`.  
The dim for each head for Multi-Head Attention.


- `pool`: string, either `cls` or `mean`, default to `mean`  
Determines if token pooling or mean pooling is applied


- `apply_rotary_emb`: bool, default `False`.  
If enabled, applies rotary_embedding in Attention blocks.

## Generative Adversarial Networks (GAN)

A generative model based on the [GAN](https://arxiv.org/abs/1406.2661) architecture.

### Usage

Since the generated images must have a certain shape, the GAN model receives
both the Generator and the Discriminator as input.

The GAN is taking care of the training process, by computing the loss and
updating the weights of the Generator and Discriminator.

Here is a code example that shows how to use the GAN interface to train on the
MNIST dataset.

```python
import torch

from torchvision import datasets, transforms

from models.generative.adverserial.gan import GAN

# Import custom Generator and Discriminator adequate to the problem
from models.generative.adverserial.mnist_discriminator import Discriminator
from models.generative.adverserial.mnist_generator import Generator 

# Create GAN
generator = Generator()
discriminator = Discriminator()
gan = GAN(generator, discriminator)

# Put model on cuda
gan.cuda()

# Create the data loader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])),
    batch_size=64,
    shuffle=True
)

# Train the GAN
gan.train_epochs(train_loader, epochs=100, print_every=100)
```

### Parameters

- `generator`: nn.Module  
The Generator model.  
Must have self.noise_dim set to the dimension of the noise vector used by the 
Generator in the forward step.


- `discriminator`: nn.Module  
The Discriminator model.
The output of the Discriminator must have shape (<batch_size, 1), having the
probability of the image being real.

### Results

This is a sample of the results of the GAN on MNIST.  

<img src="./readme_assets/fake_images.png" width="500px"></img>

For reference, this is a sample of the original MNIST dataset.  

<img src="./readme_assets/real_images.png" width="500px"></img>

### Known issues

At this moment, the gan is coded to operate only on CUDA devices.
In future the code will be refactored to allow the use of CPU devices too.