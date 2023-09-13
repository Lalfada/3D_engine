import math as M

def matrix_mul(m1, m2):
    n = len(m1[0]) # = len(m2) as it should be
    a = len(m1)
    b = len(m2[0])
    res = []
    for i in range(a):
        row = []
        for j in range(b):
            sum = 0
            for k in range(n):
                sum += m1[i][k] * m2[k][j]
            row.append(sum)
        res.append(row)
    return res


class Vec3():
    """Models a 3D vectors, and all usefull ascociated operations"""

    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __str__(self):
        return f"x: {self.x}; y: {self.y}; z: {self.z}"
    
    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other): # other is assumed to be a scalar
        return Vec3(self.x * other, self.y * other, self.z * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other): # also assumed to be a scalar
        inverse = 1.0 / other
        return self.__mul__(inverse)
    
    # __rtruediv__ doesn't exist as it wouldn't make sense

    def __pos__(self):
        return self

    def __neg__(self):
        return self.__mul__(-1.0)
    
    def __iadd__(self, other):
        return self.__add__(other)
    
    def __isub__(self, other):
        return self.__sub__(other)
    
    def __imul__(self, other):
        return self.__mul__(other)
    
    def __idiv__(self, other):
        return self.__truediv__(other)

    def length(self):
        return M.sqrt(self.x**2 
            + self.y**2
            + self.z**2)
    
    def dot(self, other):
        prod1 = self.x * other.x
        prod2 = self.y * other.y
        prod3 = self.z * other.z
        return prod1 + prod2 + prod3
        
    def cross(self, other):
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Vec3(x, y, z)
    
    def to_matrix(self):
        return [[self.x], [self.y], [self.z]]
    
    def to_extended_matrix(self):
        res = in_matrix = self.to_matrix()
        res.append([1.0])
        return res
    
    def matrix_mul(self, matrix): # a matrix is assumed to be a 3x3 list
        res_matrix = matrix_mul(matrix, self.to_matrix())
        return Vec3(res_matrix[0][0], res_matrix[1][0], res_matrix[2][0])
    
    # enables matrix multplication to add smth to the resulting vector
    def extended_matrix_mul(self, matrix): # matrix is a 3x4 list
        in_matrix = self.to_extended_matrix()
        res_matrix = matrix_mul(matrix, in_matrix)
        res_vec = Vec3(res_matrix[0][0], res_matrix[1][0], res_matrix[2][0])
        if res_matrix[3][0] != 0:
            res_vec /= res_matrix[3][0]
        return res_vec