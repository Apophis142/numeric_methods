from numpy import linspace
from math import ceil
from polynomial import Polynomial


class Function(object):
    func: callable(float)

    def __init__(self, func: callable(float)):
        self.func = func

    def __call__(self, x: float, *args, **kwargs):
        if isinstance(x, float) or isinstance(x, int):
            return self.func(x)
        else:
            raise TypeError

    def __add__(self, other):
        if isinstance(other, Function):
            if hasattr(self.func, '__add__'):
                return Function(self.func + other.func)
            return Function(lambda x: self(x) + other(x))
        if isinstance(other, float) or isinstance(other, int):
            if hasattr(self.func, '__add__'):
                return Function(self.func + other)
            return Function(lambda x: self(x) + other)

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return -1 * self

    def __mul__(self, other):
        if isinstance(other, Function):
            if hasattr(self.func, '__mul__'):
                return Function(self.func * other.func)
            return Function(lambda x: self(x) * other(x))

    def __rmul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            if hasattr(self.func, '__rmul__'):
                return Function(self.func * other)
            return Function(lambda x: self(x) * other)

    def __pow__(self, power: int, modulo=None):
        if not isinstance(power, int):
            raise TypeError
        if power == 0:
            return Function(Polynomial([1]))
        elif power < -1:
            return self ** -1 ** -power
        elif power == -1:
            return Function(lambda x: 1 / self(x) if self(x) else float("inf"))
        elif power == 1:
            return self
        elif power == 2:
            return self * self
        elif power % 2:
            return self * (self ** (power//2)) ** 2
        elif not (power % 2):
            return (self ** (power//2)) ** 2

    def __truediv__(self, other):
        return self * other**-1

    def __str__(self):
        return str(self.func)

    def derivative(self, degree: int = 1):
        if not isinstance(degree, int) or degree < 0:
            raise TypeError

        if hasattr(self.func, "derivative") and callable(getattr(self.func, "derivative")):
            return self.func.derivative(degree)

        if degree == 0:
            return self
        elif degree == 1:
            dx = .1 ** 6
            return Function(lambda x: (self(x + dx) - self(x)) / dx)
        elif degree == 2:
            dx = .1 ** 6
            return Function(lambda x: (self(x + 2*dx) - 2*self(x + dx) + self(x)) / dx**2)

    def points_of_func(self, gap: tuple[float, float], number_of_nodes: int) -> tuple[list[float, ...],
                                                                                      list[float, ...]]:
        xs, ys = [], []
        for x in linspace(*gap, number_of_nodes):
            xs += [x]
            ys += [self(x)]
        return xs, ys


class FunctionOnGap(object):
    func: Function
    gap: tuple[float, float]

    @staticmethod
    def action_on_gaps(gap1: tuple[float, float], gap2: tuple[float, float]) -> (tuple[float, float], None):
        if gap1[1] < gap2[0]:
            return None
        elif gap1[0] <= gap2[0] <= gap1[1] <= gap2[1]:
            return gap2[0], gap1[1]
        elif gap1[0] <= gap2[0] <= gap2[1] <= gap1[1]:
            return gap2
        else:
            return FunctionOnGap.action_on_gaps(gap2, gap1)

    def __init__(self, func: callable(float), gap: tuple[float, float]):
        self.func = Function(func)
        self.gap = gap if gap[0] <= gap[1] else (gap[1], gap[0])

    def __call__(self, x: float, *args, **kwargs):
        if isinstance(x, float) or isinstance(x, int):
            if self.gap[0] <= x <= self.gap[1]:
                return self.func(x)
            raise ValueError
        else:
            raise TypeError

    def __add__(self, other):
        if isinstance(other, FunctionOnGap):
            return FunctionOnGap(self.func + other.func, FunctionOnGap.action_on_gaps(self.gap, other.gap))
        if isinstance(other, Function):
            return FunctionOnGap(self.func + other, self.gap)
        if isinstance(other, float) or isinstance(other, int):
            return FunctionOnGap(self.func + other, self.gap)

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return -1 * self

    def __mul__(self, other):
        if isinstance(other, FunctionOnGap):
            return FunctionOnGap(self.func * other.func, FunctionOnGap.action_on_gaps(self.gap, other.gap))
        if isinstance(other, Function):
            return FunctionOnGap(self.func * other, self.gap)

    def __rmul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return FunctionOnGap(self.func * other, self.gap)

    def __pow__(self, power: int, modulo=None):
        if not isinstance(power, int):
            raise TypeError
        return FunctionOnGap(self.func ** power, self.gap)

    def __truediv__(self, other):
        return self * other**-1

    def __str__(self):
        return f"{self.func} on {list(self.gap)}"

    def change_gap(self, gap: tuple[float, float]):
        self.gap = gap if gap[0] <= gap[1] else (gap[1], gap[0])

    def derivative(self, degree: int = 1):
        return FunctionOnGap(self.func.derivative(degree), self.gap)

    def len(self) -> float:
        return self.gap[1] - self.gap[0]

    def points_of_func(self, number_of_nodes: int) -> tuple[list[float, ...], list[float, ...]]:
        return self.func.points_of_func(self.gap, number_of_nodes)


class FunctionSequence(object):
    func_seq: tuple[FunctionOnGap, ...]

    def __init__(self, func_seq: tuple[FunctionOnGap, ...]):
        for j in range(len(func_seq) - 1):
            if func_seq[j].gap[1] != func_seq[j+1].gap[0]:
                raise TypeError

        self.func_seq = func_seq

    def __call__(self, x: float):
        for func in self.func_seq:
            if func.gap[0] <= x <= func.gap[1]:
                return func(x)
        raise ValueError

    def derivative(self, degree: int = 1):
        return FunctionSequence(tuple(func.derivative(degree) for func in self.func_seq))

    def len(self) -> float:
        return self.func_seq[-1].gap[1] - self.func_seq[0].gap[0]

    def points_of_func(self, number_of_nodes) -> tuple[tuple[list[float, ...], list[float, ...]], ...]:
        elementary_number_of_nodes = number_of_nodes / self.len()
        res = []
        for func in self.func_seq:
            res += [func.points_of_func(ceil(elementary_number_of_nodes * func.len()))]

        return tuple(res)

    def __str__(self):
        res = ""
        for func in self.func_seq:
            res += f"{func}\n"
        return res
