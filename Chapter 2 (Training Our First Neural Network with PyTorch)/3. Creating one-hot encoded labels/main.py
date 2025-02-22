import numpy as np
import torch
import torch.nn.functional as F

y = 1
num_classes = 3

# Create the one-hot encoded vector using NumPy
one_hot_numpy = np.array([0, 1, 0])

# Create the one-hot encoded vector using PyTorch
y_tensor = torch.tensor(y)
one_hot_pytorch = F.one_hot(y_tensor, num_classes=num_classes)