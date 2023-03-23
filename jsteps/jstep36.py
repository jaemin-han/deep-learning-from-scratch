"""douple backprop

"""
if "__file__" in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)
gx = x.grad
print(gx)
x.cleargrad()

z = gx ** 3 + y
z.backward()
print(x.grad)

"""
보충 학습으로 뉴턴 방법에 대해 설명함. 아직까지는 스칼라 함수에 대해서만 다루었지만 
딥러닝은 필연적으로 많은 인자를 가지는 텐서를 입력으로 받음. 이러한 함수에 대해서 뉴턴 방법을 적용하기 위해서는
Hessian matrix를 구하고, 그것의 역행렬을 구해야 한다. 하지만 계산량이 너무 지나치게 많기에 보통 기울기만을 사용하는 방법을
딥러닝에서는 사용해 왔다.

물론 뉴턴 방법이 빨리 수렴하기는 하니까, 기울기만을 이용해서 hessian 행렬의 역행렬의 근사를 구하려는 시도가 몇몇 있다. 
또한 해세 행렬 자체는 구하기 어렵지만, 해세 행렬과 벡터의 곱은 식을 조금 변경하여 벡터와 기울기의 곱의 그래디언트로 나타낼 수 있다.

나 또한 딥러닝을 공부하면서 이차 미분을 사용하는 것은 처음 보는 것 같다. 물리학 수업 시간 때 뉴턴 방법을 배운 것 같긴 하지만..
알아둬서 나쁠 것은 없어 보인다.
"""