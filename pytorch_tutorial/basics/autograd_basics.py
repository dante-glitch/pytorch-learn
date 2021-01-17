"""
PyTorch Tensors to fit a third order polynomial to sine function

Polynomial: a + b*x + c*x^2 + d*x^3
"""

# -*- coding: utf-8 -*-

import torch
import math

# define tensor datatype and device

dtype = torch.float
device = torch.device("cpu")

# device = torch.device("cuda:0") # Uncomment to run on GPU

# Create random input vector and output

x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Random weight initialization

a = torch.rand((), device=device, dtype=dtype, requires_grad=True)
b = torch.rand((), device=device, dtype=dtype, requires_grad=True)
c = torch.rand((), device=device, dtype=dtype, requires_grad=True)
d = torch.rand((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

for t in range(2000):
    # forward pass
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # compute loss
    loss = (y_pred - y).pow(2).sum()

    if t % 100 == 0:
        print(f"loss: {loss.item()} at iteration: {t}")

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.

    loss.backward()

    with torch.no_grad():
        a -=learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

         #manually setting gradients to zero
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')




