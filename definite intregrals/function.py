from interpolation.polynomial import Polynomial
from functools import cache


NUMBERS = (int, float)


class Function(object):
    func: callable(NUMBERS)

    def __init__(self, func: callable(NUMBERS)):
        self.func = func

    @cache
    def __call__(self, x: (*NUMBERS, callable(NUMBERS))) -> (*NUMBERS, callable(NUMBERS)):
        if any(isinstance(x, tp) for tp in NUMBERS):
            return self.func(x)
        if isinstance(x, Function):
            return Function(lambda t: self.func(x(t)))

        raise TypeError

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        if isinstance(other, Function):
            if isinstance(self.func, other.func) and hasattr(self.func, '__add__'):
                return Function(self.func + other.func)
            return Function(lambda x: self(x) + other(x))
        if any(isinstance(other, tp) for tp in NUMBERS):
            if hasattr(self.func, '__add__'):
                return Function(self.func + other)
            return Function(lambda x: self(x) + other)

        raise TypeError

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if isinstance(self.func, other.func) and isinstance(other, Function):
            if hasattr(self.func, '__mul__'):
                return Function(self.func * other.func)
            return Function(lambda x: self(x) * other(x))

        raise TypeError

    def __rmul__(self, other):
        if any(isinstance(other, tp) for tp in NUMBERS):
            if hasattr(self.func, '__rmul__'):
                return Function(self.func * other)
            return Function(lambda x: self(x) * other)

        raise TypeError

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

    def derivative(self, degree: int = 1):
        if not isinstance(degree, int) or degree < 0:
            raise TypeError

        if hasattr(self.func, "derivative") and callable(getattr(self.func, "derivative")):
            return self.func.derivative(degree)

        if degree == 0:
            return self
        elif degree == 1:
            dx = 1e-6
            return Function(lambda x: (self(x + dx) - self(x)) / dx)
        elif degree == 2:
            dx = 1e-6
            return Function(lambda x: (self(x + 2*dx) - 2*self(x + dx) + self(x)) / dx**2)
