import numpy as np
y_pred = np.array([1,2,3,4,55,0,44,0])
print(np.where(y_pred == 0, -1, 1))
# print(y_pred1)