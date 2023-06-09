import numpy as np
from dezero.core import Function
from dezero.core import as_variable
from dezero import utils

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx
    
def sin(x):
    f = Sin()
    return f(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx
    
def cos(x):
    f = Cos()
    return f(x)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        # gx = gy * (1 - y * y)
        gx = gy * (1 - y ** 2)
        return gx

def tanh(x):
    f = Tanh()
    return f(x)

class Reshape(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)
    
def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

# class Transpose(Function):
#     def forward(self, x):
#         y = np.transpose(x)
#         return y
    
#     def backward(self, gy):
#         gx = transpose(gy)
#         return gx
    
# def transpose(x):
#     return Transpose()(x)


class Transpose(Function):
    def __init__(self, axes=None) -> None:
        self.axes = axes
    
    def forward(self, x): 
        y = np.transpose(x, axes=self.axes)
        return y
    
    # 내가 만든 버전
    # def backward(self, gy):
    #     if self.axes is not None:
    #         back_axes = list(self.axes)
    #         for i, axis in enumerate(self.axes):
    #             back_axes[axis] = i
    #         gx = transpose(gy, axes=back_axes)
    #         return gx
    #     else:
    #         gx = transpose(gy)
    #         return gx

    # 교제 버전
    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)    
            

def transpose(x, axes=None):
    return Transpose(axes=axes)(x)

class Sum(Function):
    def __init__(self, axis, keepdims) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = dezero.cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)