import numpy as np
from core.layers import Layer
from core.optimzer import AdamW

class Network:
    def __init__(self,architecture):
        self.arch: list[Layer] = architecture
        self.loss = None
        self.loss_d = None
        self.outputs = []
        self.loss_mag  = 0
        
        self.adamw_tracker = {}
    
    def run(self,X):
        current = X
        for layer in self.arch:
            current = layer.run(current)
        return current
    
    def tracked_run(self,X):
        self.outputs = [X]
        current = X
        for layer in self.arch:
            current = layer.run(current)
            self.outputs.append(current)
        return self.outputs

    def loss_to(self,Y):
        self.loss_mag = self.loss(Y,self.outputs[-1])
        return self.loss_mag
    
    def backprop(self, Y):
        grad = self.loss_d(self.outputs[-1], Y)  # dL/dOutput
        # loop backwards through layers
        for i in reversed(range(len(self.arch))):
            layer = self.arch[i]
            x_input = self.outputs[i]  # input to this layer
            if layer.type == "lin":
                grad = self.adamw_tracker[i].backprop(layer,x_input,grad)
            elif layer.type == "act":
                grad = grad*layer.derivative_func(x_input)

    def InitOptimizer(self,lr=0.01):
        for i,layer in enumerate( self.arch ):
            if layer.type == "lin":
                self.adamw_tracker[i] = AdamW(lr,layer.m.shape,layer.b.shape)

    def MSELoss(self):
        self.loss = lambda yt,yp: np.pow(yt-yp,2).mean()
        self.loss_d = lambda yt,yp : 2*(yt-yp)