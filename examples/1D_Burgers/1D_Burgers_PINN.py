import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(2, 50)  # Input: (x, t)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 1)  # Output: u(x, t)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = torch.tanh(self.fc1(inputs))
        out = torch.tanh(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        out = self.fc4(out)
        return out

# Define the loss function
def pinn_loss(model, x, t, u_true, nu=0.01):
    u_pred = model(x, t)
    du_dx = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    du_dt = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
    residual = du_dt + u_pred * du_dx - nu * d2u_dx2  # Burgers' equation residual
    data_loss = torch.mean((u_pred - u_true)**2)  # Data loss
    physics_loss = torch.mean(residual**2)  # Physics loss
    return data_loss + physics_loss

# Generate synthetic data
def generate_data(nu=0.01, nx=100, nt=100, L=2.0, T_max=1.0):
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T_max, nt)
    X, T = np.meshgrid(x, t)
    X = X.reshape(-1, 1)
    T = T.reshape(-1, 1)
    U = np.sin(np.pi * X) * np.exp(-nu * np.pi**2 * T)  # Analytical solution for Burgers' equation
    return X, T, U, nx, nt

# Generate data
X, T, U, nx, nt = generate_data()

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
T_tensor = torch.tensor(T, dtype=torch.float32, requires_grad=True)
U_tensor = torch.tensor(U, dtype=torch.float32)

# Initialize the model
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train the model
for it in range(10000):
    optimizer.zero_grad()
    loss = pinn_loss(model, X_tensor, T_tensor, U_tensor)
    loss.backward()
    optimizer.step()

    if it % 1000 == 0:
        print(f"Iter {it:5d}, Loss: {loss.item():.4e}")

# Predict u(x, t) using the trained model
with torch.no_grad():
    u_pred = model(X_tensor, T_tensor).numpy()

# Reshape for plotting
X_plot = X.reshape(nx, nt)
T_plot = T.reshape(nx, nt)
U_plot = u_pred.reshape(nx, nt)

# Plot the results
plt.figure(figsize=(10, 6))
plt.contourf(X_plot, T_plot, U_plot, levels=50, cmap="viridis")
plt.colorbar(label="u(x, t)")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Predicted Solution to 1D Burgers' Equation")
plt.show()
