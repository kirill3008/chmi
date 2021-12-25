import numpy as np
from copy import deepcopy


matrix1 = np.array([[3, -2, 2, -2],
                   [2, -1, 2, 0],
                   [2, 1, 4, 8],
                   [1, 3, -6, 2]],
                    dtype = np.longdouble)
r1 = np.array([8, 4, -1, 3], dtype = np.longdouble)


matrix2 = np.array([[2, 3, 1, 1],
                   [4, 3, 1, 1],
                   [1, -7, -1, -2],
                   [2, 5, 1, 1]],
                   dtype = np.longdouble)
r2 = np.array([4, 5, 7, 1], dtype = np.longdouble)
matrix = deepcopy(matrix1)
r = deepcopy(r1)

print(matrix)
print(r)



def iteration_down(matrix, r, i):   
    k = i + np.argmax(matrix[i:, i], axis = 0)
    matrix[[k,i]] = matrix[[i,k]]
    r[k], r[i] = r[i], r[k]
    print("Before \n", matrix)
    print("Swapped {} and {}".format(i,k))
    print("After \n", matrix)
    tmp = matrix[i,i]
    matrix[i] = matrix[i] / tmp
    r[i] = r[i] / tmp
    for j in range(i + 1, r.shape[0]):
        r[j] -= r[i] * matrix[j, i]
        matrix[j] -= matrix[i] * matrix[j, i]
def iteration_up(matrix, r, i):
    tmp = matrix[i][i]
    matrix[i] = matrix[i] / tmp
    for j in range(0, i):
        r[j] -= r[i] * matrix[j][i]
        matrix[j] -= matrix[i] * matrix[j][i]


for i in range(r.shape[0]):
    iteration_down(matrix, r, i)
    print(matrix)
    print(r)
for i in range(r.shape[0] -1, -1, -1):
    iteration_up(matrix, r, i)
    print(matrix)
    print(r)



print(matrix1 @ r - r1)



