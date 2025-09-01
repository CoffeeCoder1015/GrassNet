import numpy as np

from core.layers import Linear

class AdamW:
    def __init__(self,lr,m_shape,b_shape):
        self.lr = lr
        self.clip_grad = 1  # e.g., 1.0 to clip gradients
        self.weight_decay = 0.01

        # AdamW parameters
        self.m_t = np.zeros(shape=m_shape)
        self.v_t = np.zeros(shape=m_shape)
        self.mb_t = np.zeros(shape=b_shape)
        self.vb_t = np.zeros(shape=b_shape)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 1  # timestep
    
    def backprop(self,layer:Linear,layer_input,grad_output):
        # Compute gradients
        grad_w = (
            np.dot(layer_input.T, grad_output) / layer_input.shape[0]
        )  # average over batch
        grad_b = grad_output.mean(axis=0)

        # Optional gradient clipping
        if self.clip_grad is not None:
            grad_w = np.clip(grad_w, -self.clip_grad, self.clip_grad)
            grad_b = np.clip(grad_b, -self.clip_grad, self.clip_grad)

        grad_input = np.dot(grad_output, layer.m.T)

        # Update biased first and second moment estimates
        self.m_t = self.beta1 * self.m_t + (1 - self.beta1) * grad_w
        self.v_t = self.beta2 * self.v_t + (1 - self.beta2) * (grad_w**2)

        self.mb_t = self.beta1 * self.mb_t + (1 - self.beta1) * grad_b
        self.vb_t = self.beta2 * self.vb_t + (1 - self.beta2) * (grad_b**2)

        # Bias correction
        m_hat = self.m_t / (1 - self.beta1**self.t)
        v_hat = self.v_t / (1 - self.beta2**self.t)

        mb_hat = self.mb_t / (1 - self.beta1**self.t)
        vb_hat = self.vb_t / (1 - self.beta2**self.t)

        # update parameters
        layer.m -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

        # decoupled weight decay
        if self.weight_decay > 0:
            layer.m -= self.lr * self.weight_decay * layer.m
            layer.b -= self.lr * self.weight_decay * layer.b

        # Increment timestep
        self.t += 1

        return grad_input
