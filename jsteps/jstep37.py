"""reshape
원소별로 계산하는 함수는 딱히 문제가 없지만, 원소별로 계산하지 않는 함수는 고차원일 경우를 고려해야 한다.
"""

if "__file__" in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
import dezero.functions as F

# x = Variable(np.array([[1,2,3],[4,5,6]]))
# y = F.reshape(x, (6,))
# y.backward(retain_grad=False)
# print(x.grad)
# print(y)

# x = Variable(np.random.randn(1, 2, 3))
# y = x.reshape((2, 3))
# print(y)
# y = x.reshape(2, 3)
# print(y)

# x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = F.transpose(x)
# y.backward()
# print(x.grad)

# x = Variable(np.random.rand(2, 3))
# y = x.transpose()
# print(y)
# y = x.T
# print(y)

x = Variable(np.random.rand(2, 3, 4, 5))
y = x.transpose((2, 3, 1, 0))
print(y.shape)
y.backward()
print(x.grad.shape)