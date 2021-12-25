import numpy as np
from copy import deepcopy



class System(object):
    def __init__(self, matrix, b = None):
        self.matrix = matrix
        self.b = b if type(b) != None else np.zeros((matrix.shape[0]), dtype = np.longdouble)
        self.det = None
        self.rev = None



    @staticmethod
    def big_matrix(x, n, m):
        q = 1.001 - 2.0 * m * 0.001
        matrix = np.zeros((n,n), dtype = np.longdouble)
        r = np.zeros((n), dtype = np.longdouble)
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i - 1, j - 1] = np.power(q - 1, i + j)
                else:
                    matrix[i - 1, j - 1] = np.power(q, i + j) + 0.1 * (j - i)


        for i in range(1, n + 1):
            r[i - 1] = n * np.exp(x / i) * np.cos(x)
        return System(matrix, r)


    @staticmethod
    def from_examples(n):
        matrices = []
        r = []
        matrices.append(np.array([[3, -2, 2, -2], [2, -1, 2, 0], [2, 1, 4, 8],[1, 3, -6, 2]],
                    dtype = np.longdouble))
        r.append(np.array([8, 4, -1, 3], dtype = np.longdouble))
        matrices.append(np.array([[2, 3, 1, 1], [4, 3, 1, 1], [1, -7, -1, -2], [2, 5, 1, 1]],
                    dtype = np.longdouble))
        r.append(np.array([4, 5, 7, 1], dtype = np.longdouble))
        matrices.append(np.array([[1, -1, 1, -1], [4, -1, 0, -1], [2, 1, -2, 1], [5, 1, 0, -4]],
                    dtype = np.longdouble))
        r.append(np.zeros((4), dtype = np.longdouble))
        return System(matrices[n - 1], r[n - 1])




    def gauss(self):
        def iteration_down(matrix, r, i, rev):
            tmp = matrix[i][i]
            matrix[i] /=     tmp
            r[i] /= tmp
            rev[i] /= tmp
            for j in range(i + 1, r.shape[0]):
                r[j] -= r[i] * matrix[j, i]
                rev[j] -= rev[i] * matrix[j, i]
                matrix[j] -= matrix[i] * matrix[j, i]
                
            return tmp

        def iteration_up(matrix, r, i, rev):
            for j in range(0, i):
                r[j] -= r[i] * matrix[j][i]
                rev[j] -= rev[i] * matrix[j, i]
                matrix[j] -= matrix[i] * matrix[j, i]
                

        mul = 1


        matrix = deepcopy(self.matrix)
        r = deepcopy(self.b)
        rev = np.eye(self.matrix.shape[0])


        for i in range(r.shape[0]):
            mul *= iteration_down(matrix, r, i, rev)
        for i in range(r.shape[0] -1, -1, -1):
            iteration_up(matrix, r, i, rev)


        self.rev = rev
        self.x = r
        self.det = mul


    def gauss_modified(self):
        def iteration_down(matrix, r, i, rev):   
            k = i + np.argmax(matrix[i:, i], axis = 0)
            matrix[[k,i]] = matrix[[i,k]]
            rev[[k,i]] = rev[[i,k]]
            r[k], r[i] = r[i], r[k]
            tmp = matrix[i,i]
            matrix[i] /= tmp
            rev[i] /= tmp
            r[i] /= tmp
            for j in range(i + 1, r.shape[0]):
                r[j] -= r[i] * matrix[j, i]
                rev[j] -= rev[i] * matrix[j, i]
                matrix[j] -= matrix[i] * matrix[j, i]
            return tmp

        def iteration_up(matrix, r, i, rev):
            for j in range(0, i):
                r[j] -= r[i] * matrix[j][i]
                rev[j] -= rev[i] * matrix[j][i]
                matrix[j] -= matrix[i] * matrix[j][i]

        mul = 1
        matrix = deepcopy(self.matrix)
        r = deepcopy(self.b)
        rev = np.eye(self.matrix.shape[0])

        for i in range(r.shape[0]):
            mul *= iteration_down(matrix, r, i, rev)
        for i in range(r.shape[0] -1, -1, -1):
            iteration_up(matrix, r, i, rev)



        self.rev = rev
        self.x = r
        self.det = mul
    

    def get_det(self):
        if self.det:
            return self.det
        self.gauss()
        return self.det  

    def get_rev(self):
        if type(self.rev) != type(None):
            return self.rev
        self.gauss_modified()
        return self.rev 




sys = System.from_examples(1)

sys.gauss()
print(sys.det)

sys.gauss_modified()
print(sys.det)






