import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
from core.layers import Linear, Softplus
from core.network import Network
from data_tools import get_batches, init_heatmap, update_heatmap

# --- Dataset ---
X = np.linspace(-10, 10, 1000).reshape(-1, 1)
Y = np.sin(X)

# --- Network ---
n = Network([
    Linear(1, 16),
    Softplus(),
    Linear(16, 16),  # Layer 2
    Softplus(),
    Linear(16, 16),  # Layer 4
    Softplus(),
    Linear(16, 1)
])
n.MSELoss()
n.InitOptimizer()

predictions_over_epochs = []  # store predictions per epoch
layer_history1 = [n.arch[2].m.copy()]  # layer 2
layer_history2 = [n.arch[4].m.copy()]  # layer 4

# --- Training loop ---
batch_size = 200
epochs = 550

for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_Y in get_batches(X, Y, batch_size):
        n.tracked_run(batch_X)
        batch_loss = n.loss_to(batch_Y)
        n.backprop(batch_Y)
        epoch_loss += batch_loss * len(batch_X)
    
    epoch_loss /= len(X)
    print(f"Epoch {epoch}: Loss = {epoch_loss:.6f}")
    
    # Save predictions for this epoch
    pred_Y = np.array([n.run(x.reshape(1,-1)) for x in X])
    predictions_over_epochs.append(pred_Y)
    layer_history1.append(n.arch[2].m.copy())
    layer_history2.append(n.arch[4].m.copy())

# --- Animation layout ---
fig = plt.figure(figsize=(24, 16))  # taller figure
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 4], hspace=0.15, wspace=0.1)  
# top row same height as bottom now, hspace adds vertical padding

ax_pred = fig.add_subplot(gs[0, :])  # top spans both columns
axh1 = fig.add_subplot(gs[1, 0])     # bottom left
axh2 = fig.add_subplot(gs[1, 1])     # bottom right

# --- Prediction plot ---
ax_pred.plot(X, Y, color='blue', label='True function')
line, = ax_pred.plot(X, np.zeros_like(Y), color='orange', label='Prediction')
ax_pred.set_ylim(-1.5, 1.5)
ax_pred.set_title("Predictions over epochs")
ax_pred.legend()

# --- Heatmaps ---
axh1.set_title("Layer 2 (2nd Linear)")
axh2.set_title("Layer 4 (3rd Linear)")
heatmap_cax1, a1 = init_heatmap(axh1, layer_history1[0])
heatmap_cax2, a2 = init_heatmap(axh2, layer_history2[0])

# --- Animation function ---
def update(frame):
    line.set_ydata(predictions_over_epochs[frame])
    update_heatmap(layer_history1[frame], heatmap_cax1, a1)
    update_heatmap(layer_history2[frame], heatmap_cax2, a2)
    return [line, heatmap_cax1, heatmap_cax2]

ani = FuncAnimation(fig, update, frames=epochs, interval=10, blit=True)

ani.save("visualization.mp4")