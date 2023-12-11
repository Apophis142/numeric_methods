from polynomial import Polynomial


def interpolation_polynomial(table: list[tuple[(int, float), ...], ...]) -> Polynomial:
    def lagrange_multiplier(_table: list[tuple[(int, float), ...], ...], _k: int) -> Polynomial:
        res = Polynomial([1])
        x_k = _table[_k][0]
        for j in range(len(_table)):
            if j != _k:
                x_j = _table[j][0]
                res *= (Polynomial([1, -x_j]) / (x_k - x_j))
        return res

    polynomial = Polynomial([0])
    for k in range(len(table)):
        polynomial += table[k][1] * lagrange_multiplier(table, k)

    return polynomial
