import numpy
from math import pi, cos, acos, sqrt
from function import Function
from interpolation.polynomial import Polynomial


GL_alpha: float = 1 / 3
GL_beta: float = 0
GL_a: float = 1.5
GL_b: float = 3.3


def left_rectangle_formula(f: Function, a: float, b: float) -> float:
    return (b - a) * f(a)


def right_rectangle_formula(f: Function, a: float, b: float) -> float:
    return (b - a) * f(b)


def middle_rectangle_formula(f: Function, a: float, b: float) -> float:
    return (b - a) * f((a + b) / 2)


def trapezoid_formula(f: Function, a: float, b: float) -> float:
    return 1/2 * (b - a) * (f(a) + f(b))


def simpson_formula(f: Function, a: float, b: float) -> float:
    return 1/6 * (b - a) * (f(a) + 4 * f((a + b) / 2) + f(b))


def count_moments(x1: float, x2: float, p_alpha: float, p_beta: float, j: int) -> float:
    if p_alpha:
        p = j - p_alpha + 1
        return 1 / p * (x2 ** p - x1 ** p)
    elif p_beta:
        p = j - p_beta + 1
        return 1 / p * (x1 ** p - x2 ** p)


def newton_kotes_formula(f: Function, x1: float, x2: float) -> float:
    if GL_alpha:
        f = f(Function(Polynomial([1, GL_a])))
        x1 = x1 - GL_a
        x2 = x2 - GL_a
    elif GL_beta:
        f = f(Function(Polynomial([-1, GL_b])))
        x1 = GL_b - x1
        x2 = GL_b - x2

    x3 = (x1 + x2) / 2
    x = [x1, x3, x2]
    matrix_left = numpy.array([[1, 1, 1], x, [y ** 2 for y in x]])
    v_free_coefficients = [count_moments(x1, x2, GL_alpha, GL_beta, j) for j in (0, 1, 2)]
    quadrature_coefficients = list(numpy.linalg.solve(matrix_left, v_free_coefficients))

    return sum(quadrature_coefficients[j] * f(x[j]) for j in (0, 1, 2))


def cardano(pol: Polynomial) -> tuple[float, ...]:
    a, b, c, d = pol.coefficients
    q = b ** 3 / (27 * a ** 3) - b * c / (6 * a ** 2) + d / (2 * a)
    p = (3 * a * c - b ** 2) / (9 * a ** 2)
    r = sqrt(abs(p)) * (-1 + 2 * (q > 0))
    phi = acos(q / r ** 3)

    x0 = b / (3 * a)
    return  tuple(sorted((
        -2 * r * cos(phi / 3) - x0,
        2 * r * cos(pi / 3 - phi /3) - x0,
        2 * r * cos(pi / 3 + phi /3) - x0
    )))


def gauss_formula(f: Function, x1: float, x2: float) -> float:
    if GL_alpha:
        f = f(Function(Polynomial([1, GL_a])))
        x1 = x1 - GL_a
        x2 = x2 - GL_a
    elif GL_beta:
        f = f(Function(Polynomial([-1, GL_b])))
        x1 = GL_b - x1
        x2 = GL_b - x2

    moments = [count_moments(x1, x2, GL_alpha, GL_beta, j) for j in range(6)]
    matrix_left = numpy.array([moments[j:j+3] for j in range(3)])
    v_free_coefficients = [- x for x in moments[3:6]]
    p = Polynomial([1, *reversed(list(numpy.linalg.solve(matrix_left, v_free_coefficients)))])

    x = cardano(p)
    matrix_left = numpy.array([[1, 1, 1], x, [y ** 2 for y in x]])
    v_free_coefficients = numpy.array(moments[0:3])
    quadrature_coefficients = list(numpy.linalg.solve(matrix_left, v_free_coefficients))

    return sum(quadrature_coefficients[j] * f(x[j]) for j in (0, 1, 2))


def float_range(begin: float, end: float, n_steps: int) -> list[float, ...]:
   step = (end - begin) / n_steps
   return [begin + j * step for j in range(n_steps + 1)]


def integrate(f: Function, a: float, b: float, n: int, method: callable([Function, float, float])) -> float:
    res = 0
    partition = float_range(a, b, n)

    for i in range(n):
        res += method(f, partition[i], partition[i+1])
    return res
