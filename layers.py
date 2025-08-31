from abc import abstractmethod
import numpy as np

class Layer:
    def __init__(self, layer_type):
        self.type = layer_type
        self.derivative_func : any | None = None

    @abstractmethod
    def run(self,x: np.ndarray) -> np.ndarray: ...

class Activation(Layer):
    def __init__(self,function,deriv_function):
        super().__init__("act")
        self.f = function
        self.derivative_func = deriv_function
    
    def run(self, x):
        return self.f(x)

    def backprop(self, grad_output, layer_input):
        return grad_output * self.derivative_func(layer_input)
