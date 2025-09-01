from core.layers import Linear, Relu, Softplus
from core.network import Network
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

from data_tools import get_batches

input_length = 5
n = Network([
    Linear(input_length,16),
    Softplus(),
    Linear(16,16),
    Softplus(),
    Linear(16,16),
    Softplus(),
    Linear(16,1)
])   
n.MSELoss()
n.InitOptimizer()

ref_X = np.linspace(-100,100,1000)
ref_Y = np.sin(ref_X)

plt.plot(ref_X,ref_Y)

train_data = sliding_window_view(ref_Y,input_length+1)
train_x = train_data[:,:input_length]
train_y = train_data[:,input_length:]
nX = train_x.reshape(-1,input_length)
nY = train_y.reshape(-1,1)
print(nX.shape,nY.shape)

# Training loop with mini-batches
batch_size = 32
epochs = 30

for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_Y in get_batches(nX, nY, batch_size):
        n.tracked_run(batch_X)
        batch_loss = n.loss_to(batch_Y)
        n.backprop(batch_Y)
        epoch_loss += batch_loss * len(batch_X)  # sum of MSE over batch
    
    epoch_loss /= len(nX)  # average over all samples
    print(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")

start_seq = ref_Y[:input_length]
for i in range(1000-input_length):
    result = n.run(start_seq[-input_length:])
    start_seq = np.append(start_seq,result)

plt.plot(ref_X,start_seq)
plt.show()