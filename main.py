from core.layers import Linear, Softplus
from core.network import Network
import numpy as np
import matplotlib.pyplot as plt

n = Network([
    Linear(1,64),
    Softplus(),
    Linear(64,64),
    Softplus(),
    Linear(64,1)
])   
n.MSELoss()

# Example 1D dataset
X = np.linspace(-10,10,200)
Y = np.tan(X)
nX = X.reshape(-1,1)
nY = Y.reshape(-1,1)

# Batch generator
def get_batches(X, Y, batch_size=100):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]

# Training loop with mini-batches
batch_size = 200
epochs = 5000

for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_Y in get_batches(nX, nY, batch_size):
        n.tracked_run(batch_X)
        batch_loss = n.loss_to(batch_Y)
        n.backprop(batch_Y)
        epoch_loss += batch_loss * len(batch_X)  # sum of MSE over batch
    
    epoch_loss /= len(nX)  # average over all samples
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")

plt.plot(nX,nY)

X = np.linspace(-100,100,500)
nX = X.reshape(-1,1)
pred_y = n.run(nX)
plt.plot(nX,pred_y)
plt.show()
