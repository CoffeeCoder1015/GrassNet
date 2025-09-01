from core.layers import Linear, Softplus
from core.network import Network
import numpy as np
import matplotlib.pyplot as plt
from data_tools import get_batches

# --- Define a simple regression dataset ---
X = np.linspace(-10, 10, 1000).reshape(-1, 1)  # input feature
Y = np.sin(X)                                  # target value

plt.plot(X, Y, label="True function")

# --- Define the network ---
n = Network([
    Linear(1, 16),
    Softplus(),
    Linear(16, 16),
    Softplus(),
    Linear(16, 16),
    Softplus(),
    Linear(16, 1)
])
n.MSELoss()
n.InitOptimizer()

# --- Training loop ---
batch_size = 1000
epochs = 2000

for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_Y in get_batches(X, Y, batch_size):
        n.tracked_run(batch_X)
        batch_loss = n.loss_to(batch_Y)
        n.backprop(batch_Y)
        epoch_loss += batch_loss * len(batch_X)
    
    epoch_loss /= len(X)
    print(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")

# --- Evaluate the network ---
pred_Y = np.array([n.run(x.reshape(1,-1))[0][0] for x in X])
plt.plot(X, pred_Y, label="Predictions")
plt.legend()
plt.show()
