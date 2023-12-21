import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        self.t1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        
        self.t2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.t3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        
        self.t3b = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.t4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        
        self.t4b = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.t5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
                        
        
        self.head = nn.Sequential(
            # flatten
            nn.Flatten(),
            
            # Head                        
            nn.Linear(512, num_classes)
            #nn.LogSoftmax(dim=1)
            )
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        
        out1 = self.t1(x)
        out2 = self.t2(out1)
        out3 = self.t3(out2)
        out4 = self.t3b(out3)
        out5 = self.t4(out4)
        out6 = self.t4b(out5)
        out7 = self.t5(out6)
        
        out8 = self.avgpool(out7)
        out9 = self.head(out8)        
                
        return out9


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
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
