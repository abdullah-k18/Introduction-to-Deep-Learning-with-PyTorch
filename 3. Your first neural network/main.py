import torch
import torch.nn as nn

input_tensor = torch.Tensor([[2, 3, 6, 7, 9, 3, 2, 1]])

# Implement a small neural network with two linear layers
model = nn.Sequential(nn.Linear(in_features=8, out_features=4),
                       nn.Linear(in_features=4, out_features=1)
                     )

output = model(input_tensor)
print(output)