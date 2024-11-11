import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()

        self.conv2d = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2d2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2d3 = nn.Conv2d(64, 128, 3, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.output = nn.Linear(1152, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv2d(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2d2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2d3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.output(x)
        return x


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
