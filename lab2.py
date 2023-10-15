import matplotlib.pyplot as plt


class Matrix(object):
    """"""

    def __init__(self, args: [list[list[int, float, complex]], tuple[tuple[int, float, complex]]]):
        cols = len(max(args, key=lambda x: len(x)))
        self.matrix = tuple(tuple(row[i] if i < len(row) else 0 for i in range(cols)) for row in args)
        self.cols = cols
        self.rows = len(args)

    def __repr__(self):
        return str(self.matrix)

    def __str__(self):
        cols_length = [max(len(str(el)) for el in
                           [self.matrix[j][i] for j in range(self.rows)]
                           ) for i in range(self.cols)]
        return "(" + ")\n(".join([" ".join([str(row[i]).rjust(cols_length[i]) for i in range(self.cols)])
                                  for row in self.matrix]) + ")"

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError

        return Matrix([
            [self.matrix[i][j] + other.matrix[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __abs__(self) -> int:
        return max(sum(map(abs, row)) for row in self.matrix)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError

        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return Matrix([[el * other for el in row] for row in self.matrix])
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError
            return Matrix([
                [sum(self.matrix[r][j] * other.matrix[j][c] for j in range(self.cols))
                 for c in range(other.cols)]
                for r in range(self.rows)
            ])
        raise TypeError

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power: int, modulo=None):
        if not isinstance(power, int):
            raise TypeError
        if self.cols != self.rows:
            raise ValueError
        if power == -1:
            return self.reverse()
        if power == 0:
            return Matrix([
                [int(i == j) for i in range(self.cols)]
                for j in range(self.rows)
            ])
        if power == 1:
            return self
        if power == 2:
            return self * self
        if not power % 2:
            return (self ** (power // 2)) ** 2
        if power % 2:
            return (self ** (power // 2)) ** 2 * self

    def __getitem__(self, item):
        return self.matrix[item]

    def to_tuple(self):
        if self.rows == 1:
            return tuple(self[0])
        if self.cols == 1:
            return tuple(self.t()[0])
        raise TypeError

    def t(self):
        return Matrix([
            [self.matrix[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def det(self) -> [int, float, complex]:
        if self.cols != self.rows:
            raise ValueError
        if self.cols == 1:
            return self.matrix[0][0]
        return sum(((-1)**k * self.matrix[0][k] * self.adding(0, k).det() for k in range(self.rows)))

    def reverse(self):
        if (det := self.det()) == 0:
            raise ValueError
        return Matrix([
            [(-1) ** (i+j) * self.adding(i, j).det() for j in range(self.cols)]
            for i in range(self.rows)
        ]).t() * (1 / det)

    def adding(self, n_row, n_col):
        return Matrix([
            [self.matrix[i][j] for j in range(self.cols) if j != n_col]
            for i in range(self.rows) if i != n_row
        ])

    def insert_col(self, col, col_num: int):
        if not isinstance(col, Matrix):
            raise TypeError
        if self.rows != col.rows or not 0 < col_num <= self.cols:
            raise IndexError

        return Matrix([
            [(self.matrix[i][j] if j != col_num - 1 else col.matrix[i][0]) for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def mul_row(self, n_row: int, c: int):
        if c == 1:
            return self
        q = Matrix([
            [int(i == j) * ((c - 1) * (n_row == i) + 1) for j in range(self.rows)]
            for i in range(self.rows)
        ])

        return q * self

    def add_row(self, n_row1: int, n_row2: int, c: int):
        if c == 0:
            return self
        s = Matrix([
            [int(i == j) + (c if j == n_row2 and i == n_row1 else 0) for j in range(self.rows)]
            for i in range(self.rows)
        ])

        return s * self


def get_system(n: int) -> tuple[Matrix, Matrix]:
    return Matrix([
        [n + 2, 1, 1],
        [1, n + 4, 1],
        [1, 1, n + 6]
    ]), \
           Matrix([
               [n + 4, n + 6, n + 8]
           ]).t()


def solve_system(a: Matrix, b: Matrix) -> Matrix:
    if a.rows != b.rows:
        raise IndexError
    if a.det() == 0:
        raise ZeroDivisionError

    delta = a.det()
    return Matrix([
        [a.insert_col(b, i + 1).det() / delta] for i in range(a.cols)
    ])


def mpi(a: Matrix, b: Matrix, acc: float = 1e-8) -> tuple[Matrix, int]:
    mu = 1 / abs(a)
    c = mu * b
    b = Matrix([[int(i == j) for j in range(a.cols)] for i in range(a.rows)]) - mu * a
    del a, mu
    print(abs(b))

    k = 1
    prev_x = c
    curr_x = b * prev_x + c
    while acc <= abs(abs(b) / (1 - abs(b)) * abs(curr_x - prev_x)):
        prev_x, curr_x = curr_x, b * curr_x + c
        k += 1

    return curr_x, k


def gauss_method(a: Matrix, b: Matrix) -> Matrix:
    for n in range(a.rows):
        q = 1 / a.matrix[n][n]
        a = a.mul_row(n, q)
        b = b.mul_row(n, q)
        for i in range(a.rows):
            if i == n:
                continue
            k = -a.matrix[i][n] / a.matrix[n][n]
            a = a.add_row(i, n, k)
            b = b.add_row(i, n, k)

    return b


def get_bad_system(size: int, n: int, epsilon: float) -> tuple[Matrix, Matrix]:
    return Matrix([
        [0] * i + [1] + [-1] * (size - i - 1) for i in range(size)
    ]) + n * epsilon * Matrix([
        [1] * (i + 1) + [-1] * (size - i - 1) for i in range(size)
    ]), \
           Matrix([[-1] * (size - 1) + [1]]).t()


if __name__ == "__main__":
    mat_a, vec_b = get_bad_system(10, 8, 1e-5)
    xs, iter_nums = [], []
    for acc in range(4, 16):
        xs += [acc]
        acc = 10**(-acc)
        *rest, iter_num = mpi(mat_a, vec_b, acc)
        iter_nums += [iter_num]

    plt.subplots(figsize=(10, 6))
    plt.subplot(111)
    plt.plot(xs, iter_nums, "-b.", alpha=0.9, label="кол.-во итер.")
    plt.show()

    print(gauss_method(*get_bad_system(10, 10, 1e-3)))
