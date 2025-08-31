from core.layers import Linear, Softplus
from core.network import Network


n = Network([
    Linear(1,64),
    Softplus(),
    Linear(64,64),
    Softplus(),
    Linear(64,1)
])   
n.MSELoss()
