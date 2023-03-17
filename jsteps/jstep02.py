import numpy as np

class Variable:
    def __init__(self, data) -> None:
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        """이 method는 상속해서 구현해야 하기 떄문에 raise statement가 있음"""
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
x = Variable(np.array(19))
f = Square()
y = f(x)
print(type(y))
print(y.data)

    