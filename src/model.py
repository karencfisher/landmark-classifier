import torch
import torch.nn as nn
from collections import OrderedDict


# define the CNN architecture
class ConvBlock(nn.Module):
    def __init__(self, num_conv_layers, input_channels, output_channels):
        super().__init__()
        
        layers = []        
        for i in range(num_conv_layers):
            input_channel = input_channels if i == 0 else output_channels
            layers.append((f'conv{i+1}', nn.Conv2d(input_channel, output_channels,
                                                   kernel_size=3, padding=1)))
            layers.append((f'relu{i+1}', nn.ReLU()))
        layers.append(('pool', nn.MaxPool2d(2)))
        self.convblock = nn.Sequential(OrderedDict(layers))
        
    def forward(self, x):
        return self.convblock(x)
    
        
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        self.seq_model = nn.Sequential(
            OrderedDict([
                ('convblock1', ConvBlock(2, 3, 64)),
                ('convblock2', ConvBlock(2, 64, 128)),
                ('convblock3', ConvBlock(2, 128, 256)),
                ('convblock4', ConvBlock(2, 256, 512)),
                ('convblock5', ConvBlock(2, 512, 512)),
                
                ('flatten', nn.Flatten()),
                ('fc1', nn.Linear(25088, 4096)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(4096, 4096)),
                ('relu2', nn.ReLU()),
                ('fc3', nn.Linear(4096, num_classes)),
                ('logsoftmax', nn.LogSoftmax(dim=1))
            ])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.seq_model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
