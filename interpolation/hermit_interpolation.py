from polynomial import Polynomial
from lagrange_interpolation import interpolation_polynomial as lip
from math import factorial


def cmb(n: int, k: int) -> int:
    return factorial(n) // factorial(k) // factorial(n - k)


def node_polynomial(table: list[tuple[(int, float), ...], ...]) -> Polynomial:
    res = Polynomial([1])
    for k in range(len(table)):
        res *= Polynomial([1, -table[k][0]])
    return res


def interpolation_polynomial(table: list[tuple[(int, float), ...], ...]) -> Polynomial:
    def calculate_new_nodes(_table: list[tuple[(int, float), ...], ...], p_0: Polynomial) \
            -> list[tuple[(int, float), ...], ...]:
        new_table = [[node[0]] for node in _table if len(node) > 2]
        nd_p = node_polynomial(_table)
        _table = [node for node in _table if len(node) > 2]
        for j in range(len(_table)):
            x_j = table[j][0]
            for k in range(2, len(_table[j])):
                new_table[j] += [
                    (_table[j][k] - p_0.derivative(k-1)(x_j) -
                        sum([cmb(k-1, i) * nd_p.derivative(k-i-1)(x_j) * new_table[j][i+1] for i in range(k-2)])) /
                    ((k-1) * nd_p.derivative()(x_j))
                                 ]
        return [tuple(node) for node in new_table]

    nodes = [node[0:2] for node in table]
    res = lip(nodes)
    if max(map(len, table)) == 2:
        return res
    return res + node_polynomial(nodes) * interpolation_polynomial(calculate_new_nodes(table, res))


if __name__ == "__main__":
    print(pol := interpolation_polynomial([(-2, 3, 1), (-1, 1, 2), (0, 0), (1, 2, 2), (2, -1)]),
          pol(1/2))
