import numpy as np
import random
x = np.zeros((10,5,3))
for i in range(10):
    for j in range(5):
        for k in range(3):
            x[i][j][k] = random.randint(0,5)
print('x_bef',x)
x = np.reshape(x,(10,5,3))
x = np.flip(x,0)
print('x_aft',x)

for i in range(5):
    print(x[0][i][0])

x = np.swapaxes(x,1,2)
x = np.swapaxes(x,0,1)

print("swap",x)

y = x.ravel()
print(y)
