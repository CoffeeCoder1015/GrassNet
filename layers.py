from abc import abstractmethod
import numpy as np

class Layer:
    def __init__(self, layer_type):
        self.type = layer_type
        self.derivative_func : any | None = None

    @abstractmethod
    def run(self,x: np.ndarray) -> np.ndarray: ...

class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__("lin")
        # He initialization for ReLU
        self.m = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.b = np.zeros(output_size)

    def run(self, x: np.ndarray):
        return np.dot(x, self.m) + self.b

class Activation(Layer):
    def __init__(self,function,deriv_function):
        super().__init__("act")
        self.f = function
        self.derivative_func = deriv_function
    
    def run(self, x):
        return self.f(x)

class Relu(Activation):
    def __init__(self):
        super().__init__(
            lambda x: np.where(x>0,x,0), 
            lambda x: np.where(x>0,1,0)
        )

class Tanh(Activation):
    def __init__(self):
        super().__init__(
            np.tanh,
            lambda x: 1 - np.pow(np.tanh(x),2)
        )

class LeakyRelu(Activation):
    def __init__(self, alpha=0.01):
        """
        alpha: slope for negative inputs (default 0.01)
        """
        super().__init__(
            function=lambda x: np.where(x > 0, x, alpha * x),
            deriv_function=lambda x: np.where(x > 0, 1, alpha)
        )

class Softplus(Activation):
    def __init__(self):
        super().__init__(
            lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0),  # stable softplus
            lambda x: 1 / (1 + np.exp(-x))  # derivative = sigmoid(x)
        )
