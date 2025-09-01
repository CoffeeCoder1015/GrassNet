# GrassNet 🌱

GrassNet is a **neural network library built from scratch in Python**.  
It provides a lightweight API for defining and training feedforward neural networks without relying on external frameworks like PyTorch or TensorFlow — just NumPy.  

This project is meant as both a **learning tool** and a **minimal framework** that shows how deep learning works under the hood — forward propagation, backpropagation, optimizers, and training loops.

---

https://github.com/user-attachments/assets/175e5349-a536-4fad-be6e-2dda7a88e997

<div style="text-align: center;">
    Generated via 
    <a href="visualize.py">visualize.py</a>
</div>

---

## Features
- Custom API for defining feedforward networks  
- Layers: `Linear` (Dense), activation functions (`ReLU`, `Softplus`, etc.)  
- Loss functions: Mean Squared Error (MSE)  
- Optimizer: **AdamW (decoupled weight decay)** implemented from scratch  
- Example: sequence-to-sequence prediction of `sin(x)` in `main.py`

---

## Example Usage
```python
from core.layers import Linear, Softplus
from core.network import Network
import numpy as np

# Define network
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

n.MSELoss()       # Set loss function
n.InitOptimizer() # Initialize AdamW

n.tracked_run(X)  # A single run where layer outputs are tracked
n.loss_to(Y)      # Returns the loss of the latest run's output compared to Y
n.backprop(Y)     # Do one backpropagation iteration based on Y
n.run(X)          # Standard run for inference

```
Full example found in [main.py](main.py)

`main.py` implements a *sequence-to-sequence* prediction example:

1. Generates training data from sin(x)

2. Trains the model using mini-batches and AdamW

3. Uses the trained model to iteratively predict the sine sequence

4. Plots both the original and predicted wave

There is also a variant of `main.py` found in `regression.py` which implements regression
prediction instead of *sequence-to-sequence* prediction.


## Project structure
```
.
├── core/
│   ├── layers.py     # Linear + activations
│   ├── network.py    # Network API (forward, backprop, loss, optimizer)
│   └── optim.py      # AdamW optimizer
├── data_tools.py     # Utility functions for batching
├── main.py           # Example: sin(x) sequence prediction
├── regression.py     # Same as main.py but is regression instead
└── README.md
```

## Try it out!

```bash
git clone https://github.com/CoffeeCoder1015/grassnet.git
cd grassnet
pip install numpy matplotlib
python main.py
```
