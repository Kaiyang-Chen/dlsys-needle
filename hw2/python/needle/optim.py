"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for w in self.params:
            if w not in self.u:
                # NOTE: intialized with zeros
                self.u[w] = 0
            # NOTE: param.grad maybe None
            if w.grad is None:
                continue 
            grad = ndl.Tensor(w.grad.numpy(), dtype='float32').data + self.weight_decay * w.data
            self.u[w] = self.momentum * self.u[w] + (1 - self.momentum) * grad
            w.data = w.data - self.lr * self.u[w]
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            if param not in self.m:
                self.m[param] = ndl.init.zeros(*param.shape)
                self.v[param] = ndl.init.zeros(*param.shape)
            if param.grad is None:
                continue
            grad_data = ndl.Tensor(param.grad.numpy(), dtype='float32').data \
                 + param.data * self.weight_decay
            self.m[param] = self.beta1 * self.m[param] \
                + (1 - self.beta1) * grad_data
            self.v[param] = self.beta2 * self.v[param] \
                + (1 - self.beta2) * grad_data**2
            # NOTE: bias correction
            u_hat = (self.m[param]) / (1 - self.beta1 ** self.t)
            v_hat = (self.v[param]) / (1 - self.beta2 ** self.t)
            param.data = param.data - self.lr * u_hat / (v_hat ** 0.5 + self.eps) 
        ### END YOUR SOLUTION
