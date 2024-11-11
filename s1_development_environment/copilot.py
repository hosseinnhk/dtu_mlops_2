import torch
import torch.nn as nn


class ThreeLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ThreeLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Example usage
input_size = 784  # Example input size (e.g., 28x28 images)
hidden_size1 = 128
hidden_size2 = 64
output_size = 10  # Example output size (e.g., 10 classes for classification)

model = ThreeLayerNet(input_size, hidden_size1, hidden_size2, output_size)
print(model)
