y_pred = np.array(10)
y = np.array(1)

# Calculate the MSELoss using NumPy
mse_numpy = np.mean((y_pred - y) ** 2)

# Create the MSELoss function
criterion = nn.MSELoss()

y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Calculate the MSELoss using the created loss function
mse_pytorch = criterion(y_pred_tensor, y_tensor)
print(mse_pytorch)