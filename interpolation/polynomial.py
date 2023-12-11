NUMBERS = (int, float, complex)


class Polynomial(object):
    degree: int
    coefficients: list[float, ...]

    def __call__(self, x: float, *args, **kwargs) -> float:
        if not any([isinstance(x, tp) for tp in NUMBERS]):
            raise TypeError
        res = 0
        x_p = 1
        for c in reversed(self.coefficients):
            res += c * x_p
            x_p *= x
        return res

    def __init__(self, coefficients: list[float, ...]):
        if not coefficients:
            coefficients = [0]
        while len(coefficients) - 1 and not coefficients[0]:
            del coefficients[0]
        self.coefficients = coefficients
        self.degree = len(coefficients) - 1

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        if any([isinstance(other, tp) for tp in NUMBERS]):
            return self + Polynomial([other])
        elif not isinstance(other, Polynomial):
            raise TypeError
        p1 = self.coefficients
        p2 = other.coefficients
        if self.degree < other.degree:
            degree = other.degree
            p1 = [0] * (other.degree - self.degree) + p1
        else:
            degree = self.degree
            p2 = [0] * (self.degree - other.degree) + p2

        return Polynomial([p1[k] + p2[k] for k in range(degree + 1)])

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        res = self + other
        self.coefficients, self.degree = res.coefficients, res.degree
        return self

    def __sub__(self, other):
        return self + (-other)

    def __isub__(self, other):
        res = self - other
        self.coefficients, self.degree = res.coefficients, res.degree
        return self

    def __mul__(self, other):
        if not isinstance(other, Polynomial):
            raise TypeError
        res = Polynomial([0])
        for j in range(other.degree + 1):
            p = Polynomial(self.coefficients + [0] * j)
            res += other.coefficients[-j - 1] * p
        return res

    def __rmul__(self, other):
        if any([isinstance(other, tp) for tp in NUMBERS]):
            return Polynomial([other * c for c in self.coefficients])
        raise TypeError

    def __imul__(self, other):
        res = self * other
        self.coefficients, self.degree = res.coefficients, res.degree
        return self

    def __truediv__(self, other):
        if any([isinstance(other, tp) for tp in NUMBERS]):
            return (1/other) * self
        raise TypeError

    def __itruediv__(self, other):
        res = self / other
        self.coefficients, self.degree = res.coefficients, res.degree
        return self

    def __pow__(self, power: int, modulo=None):
        if not isinstance(power, int):
            raise TypeError
        elif power < 0:
            raise TypeError

        if power == 0:
            return Polynomial([1])
        elif power == 1:
            return self
        elif power == 2:
            return self * self
        elif power % 2:
            return self * (self ** (power//2)) ** 2
        else:
            return (self ** (power//2)) ** 2

    def __str__(self):
        if self.coefficients == [0]:
            return "0"

        res = ""
        p = self.degree
        for c in self.coefficients[:-1:]:
            if c:
                res += \
                    f"{'' if c < 0 or p == self.degree else '+'}" +\
                    f"{'-' if c == -1. else '' if c == 1. else round(c, 3)}x{'^' + str(p) if p - 1 else ''}"
            p -= 1
        c = round(self.coefficients[-1], 3)
        res += f"{'+' if c > 0 and self.degree else ''}{c}" if c else ''
        return res

    def derivative(self, deg: int = 1):
        if not isinstance(deg, int) or deg < 0:
            raise TypeError
        if deg == 0:
            return self
        if deg == 1:
            coefficients = []
            for deg in range(self.degree):
                coefficients += [(self.degree - deg) * self.coefficients[deg]]
            return Polynomial(coefficients)
        return self.derivative().derivative(deg - 1)

    def indefinite_integral(self):
        coefficients = []
        i = 0
        for deg in range(self.degree + 1, 0, -1):
            coefficients += [self.coefficients[i] / deg]
            i += 1
        coefficients += [0]
        return Polynomial(coefficients)

    def definite_integral(self, a: float, b: float):
        pol = self.indefinite_integral()
        return pol(b) - pol(a)
