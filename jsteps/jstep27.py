"""
Taylor Series 
"""
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    for path in sys.path:
        print(path)

import numpy as np
from dezero import Function

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx
    
def sin(x):
    f = Sin()
    return f(x)

from dezero import Variable

# x = Variable(np.array(np.pi/4))
# y = sin(x)
# y.backward()

# print(y.data)
# print(x.grad)

import math
from dezero.utils import plot_dot_graph

def my_sin(x, threshold=1e-4):
    y = 0
    for i in range(100_000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

x = Variable(np.array(np.pi/4))
y = my_sin(x)
y.backward()

plot_dot_graph(y, to_file='my_sin.png')

print(y.data)
print(x.grad)