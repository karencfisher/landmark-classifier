import torch
import torch.nn as nn


# define the CNN architecture
# make consistent convolutional blocks as in VGG architecture
class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))

        return self.pool(x)
    
        
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        self.convblock1 = ConvBlock(3, 64)
        self.convblock2 = ConvBlock(64, 128)
        self.convblock3 = ConvBlock(128, 256)
                
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(200704, 4096)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)

        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# class MyModel(nn.Module):
#     def __init__(self, num_classes, dropout=0):
#         super(MyModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(16 * 112 * 112, num_classes)  # Adjust based on input size

#     def forward(self, x):
#         x = self.pool(nn.ReLU()(self.conv1(x)))
#         x = x.view(x.size(0), -1)  # Flatten
#         x = self.fc1(x)
#         return x


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
