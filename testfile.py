import numpy as np
from dezero import Variable

x = Variable(np.array(1.0))
y = x + 2
print(y)