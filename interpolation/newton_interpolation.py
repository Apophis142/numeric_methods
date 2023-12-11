from polynomial import Polynomial


def divided_difference(nodes: list[tuple[(int, float), ...], ...]) -> list[list[(int, float), ...], ...]:
    n = len(nodes)
    res = [[nodes[i][1]] for i in range(n)]

    for i in range(1, n):
        for j in range(n - i):
            res[j] += [(res[j+1][-1] - res[j][-1]) / (nodes[j + i][0] - nodes[j][0])]

    return res


def sequence_interpolation_polynomials(nodes: list[tuple[(int, float), ...], ...]) -> list[Polynomial, ...]:
    def node_polynomial(xs):
        res = Polynomial([1])
        for x in xs:
            res *= Polynomial([1, -x[0]])
        return res

    div_diffs = divided_difference(nodes)
    seq = [Polynomial([nodes[0][1]])]

    for k in range(1, len(nodes)):
        seq += [seq[-1] + div_diffs[0][k] * node_polynomial(nodes[:k])]

    return seq
