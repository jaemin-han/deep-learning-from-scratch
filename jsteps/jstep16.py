"""
복잡한 계산 그래프도 처리할 수 있게 함수 별 역전파 우선순위를 설정함

일반적인 계산 때 함수가 변수를 만들어내는 과정을 목격함. 이를 바탕으로 세대를 기록할 수 있음
generation 추가

"""
import numpy as np

class Variable:
    def __init__(self, data) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)} is not supported')

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        # 굳이 seen_set이 있어야 하는 이유를 모르겠다. 이게 없으면 어떤 문제가 생길 수 있지?
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        # 이 함수를 대신 사용하면 역전파를 불필요하게 여러번 하게 돼서 미분값이 늘어나게 된다.
        # def add_func2(f):
        #     funcs.append(f)
        #     funcs.sort(key=lambda x: x.generation)                
                
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

    def cleargrad(self):
        self.grad = None

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys =  self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        
        outputs = [Variable(as_array(y)) for y in ys]

        # 추가
        self.generation = max([x.generation for x in inputs])

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
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
        return gx

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
    f = Add()
    return f(x0, x1)



x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)

@profile
def for_test():
    for i in range(100):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))

for_test()