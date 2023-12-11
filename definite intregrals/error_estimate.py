import numpy
from function import Function
from quadrature_formulas import integrate
from math import log
from functools import cache


def richardson(m: float, n_partition: list[int, ...], integral_val: list[float, ...], a: float, b: float)\
        -> tuple[float, float]:
    hs = [(b - a) / j for j in n_partition]
    cycle = range(len(n_partition))
    a = numpy.array([[hs[i] ** (m + k) for k in cycle[:-1]] + [-1] for i in cycle])
    b = [-x for x in integral_val]

    *c, val = numpy.linalg.solve(a, b)
    r_h = abs(sum(c[k] * hs[k] ** (m + k) for k in cycle[:-1]))

    return val, r_h


@cache
def aitken(func: Function, method: callable([Function, float, float]), a: float, b: float, base_partition: int) -> float:
    s = [integrate(func, a, b, n, method) for n in (base_partition, base_partition * 2, base_partition * 2**2)]
    return log(abs((s[2] - s[1]) / (s[1] - s[0]))) / log(.5)
