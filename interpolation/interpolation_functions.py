import math as m


def func1(x: (float, int)) -> float:
    return m.sin(x) - x - .25


def func2(x: (float, int)) -> float:
    return x**3 - m.exp(x) + 1


def func3(x: (float, int)) -> float:
    return m.tan(.5*x + .2) - x**2


def func4(x: (float, int)) -> float:
    return 3*x - m.cos(x) - 1


def func5(x: (float, int)) -> float:
    return x**2 + 4*m.sin(x) - 2


def func6(x: (float, int)) -> float:
    return x**2 - 10*m.sin(x)


def func7(x: (float, int)) -> float:
    return .5**x + 1 - (x - 2)**2


def func8(x: (float, int)) -> float:
    return (x + 3)*m.cos(x) - 1


def func9(x: (float, int)) -> float:
    return x**2*m.cos(x) + 1


def func10(x: (float, int)) -> float:
    # return m.cos(x + .3) - x**2
    return 1 / (1 + 25 * x**2)


INTERPOLATION_FUNCTIONS = (
    func10, func1, func2, func3, func4, func5, func6, func7, func8, func9
)
