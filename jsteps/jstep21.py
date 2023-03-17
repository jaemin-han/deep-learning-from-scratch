"""operator overloading2
Variable 인스턴스를 +, * 로 코딩할 수 있게 하는게 목표
+
Variable 을 np.ndarray 와 int float 와 같이 사용할 수 있게 하자

책에서는 add, mul은 radd, rmul과 동일하게 취급하면 된다고 한다. 물론 결과는 그렇지만, 
mul의 경우는 역전파 때 두 값이 바뀌어야 하는데 문제가 없는걸까?
생각해봤는데 상관없네.. 문제생길이유가없음
"""
import numpy as np
import weakref
import contextlib

class Variable:
    __array_priority__ = 200
    def __init__(self, data, name=None) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported.')
        
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)
            
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None
            
    def cleargrad(self):
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
            return len(self.data)
    
    def __repr__(self) -> str:
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    def __rmul__(self, other):
        print('rmul excuted')
        return add(self, other)
    
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(input) for input in inputs]
        
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs):
        raise NotImplementedError()

    def backward(self, *gys):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gy
    
def square(x):
    f = Square()
    return f(x)

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
def add(x0, x1):
    x1 = as_array(x1)
    f = Add()
    return f(x0, x1)

Variable.__add__ = add
Variable.__radd__ = add

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        # 교제는 위 코드이나, 뭔가 답답해보이는 코드여서 아래로 변경함
        # 아래는 generator expression을 사용했는데, 문제가 없을까?
        x0, x1 = (input.data for input in self.inputs)
        return gy * x1, gy * x0
    
def mul(x0, x1):
    x1 = as_array(x1)
    f = Mul()
    return f(x0, x1)

Variable.__mul__ = mul
Variable.__rmul__ = mul

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy

def neg(x):
    f = Neg()
    return f(x)

Variable.__neg__ = neg

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy

def sub(x0, x1):
    x1 = as_array(x1)
    f = Sub()
    return f(x0, x1)

Variable.__sub__ = sub

def rsub(x0, x1):
    x1 = as_array(x1)
    f = Sub()
    return f(x1, x0)

Variable.__rsub__ = rsub

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = (input.data for input in self.inputs)
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    f = Div()
    return f(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    f = Div()
    return f(x1, x0)

Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv

class Pow(Function):
    def __init__(self, c) -> None:
        self.c = c
    
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
    
def pow(x, c):
    f = Pow(c)
    return f(x)

Variable.__pow__ = pow

x = Variable(np.array(2.0))
y = x ** 3
print(y)