from polynomial import Polynomial
from function_on_gap import FunctionOnGap
import matplotlib.pyplot as plt
from interpolation_functions import INTERPOLATION_FUNCTIONS as FUNCS
from interpolation import ln_diff_functions


def find_orthogonal_polynomials(points: list[float, ...], amount: int) -> list[Polynomial, ...]:
    m = len(points)
    q = [Polynomial([1]), Polynomial([1, -sum(points)/m])]

    for j in range(1, amount):
        alpha = sum([x * q[j - 1](x)**2 for x in points]) / sum([q[j - 1](x)**2 for x in points])
        beta = sum([x * q[j](x) * q[j - 1](x) for x in points]) / sum([q[j - 1](x)**2 for x in points])

        q += [Polynomial([1, -alpha]) * q[-1] - beta * q[-2]]

    return q


def find_approximate_polynomial(table: list[tuple[float, float], ...], degree: int) -> Polynomial:
    table = sorted(table, key=lambda x: x[0])
    points = [point[0] for point in table]
    q = find_orthogonal_polynomials(points, degree - 1)

    a = [sum([pol(point[0]) * point[1] for point in table]) / sum([pol(point[0])**2 for point in table]) for pol in q]

    res = Polynomial([0])
    for k in range(degree):
        res += a[k] * q[k]
    return res


def main() -> None:
    table = curr_func.points_of_func(curr_number_of_nodes)
    table = [(table[0][j], table[1][j]) for j in range(len(table[0]))]
    nodes = [[], []]
    for point in table:
        nodes[0] += [point[0]]
        nodes[1] += [point[1]]

    pol = FunctionOnGap(find_approximate_polynomial(table, curr_degree + 1), interpolation_interval)

    ax = axis[0]
    ax.clear()
    ax.set_title("Функция и аппроксимирующий полином")
    ax.plot(*pol.points_of_func(1000), '--b', alpha=.7, label=f"полином {curr_degree}й степени")
    ax.plot(*curr_func.points_of_func(1000), 'r', alpha=.4, label=f"иск. функция")
    ax.scatter(nodes[0], nodes[1], color='r', alpha=.5, s=100/curr_number_of_nodes,
               label=f"узлы ({curr_number_of_nodes})")
    ax.legend()

    ax = axis[1]
    ax.clear()
    ax.set_title("Порядок погрешности")
    error = FunctionOnGap(ln_diff_functions(curr_func, pol), interpolation_interval)
    ax.plot(*error.points_of_func(1000), '--r')

    plt.show()


def on_key_event(event) -> None:
    global curr_number_of_nodes, curr_degree
    global interpolation_interval
    global curr_func
    key = getattr(event, 'key')

    if key == 'escape':
        interpolation_interval = (-1., 1.)
        curr_number_of_nodes = 7
        curr_degree = 2
        curr_func = FunctionOnGap(FUNCS[1], interpolation_interval)
    elif key.isdigit():
        curr_func = FunctionOnGap(FUNCS[int(key)], interpolation_interval)
    elif key.startswith('ctrl+') and key[-1].isdigit():
        def tmp(x):
            return abs(x) * FUNCS[int(key[-1])](x)

        curr_func = FunctionOnGap(tmp, interpolation_interval)
    elif key in ('+', '=', 'up'):
        curr_number_of_nodes += 1
    elif key in ('-', 'down'):
        curr_number_of_nodes = max(3, curr_number_of_nodes - 1)
        curr_degree = min(curr_degree, curr_number_of_nodes - 1)
    elif key in ('ctrl+plus', 'ctrl+eq', 'ctrl+up'):
        curr_degree = min(curr_degree + 1, curr_number_of_nodes - 1)
    elif key in ('ctrl+minus', 'ctrl+down'):
        curr_degree = max(0, curr_degree - 1)
    elif key == 'right':
        interpolation_interval = (
            round(interpolation_interval[0] - .1, 6),
            round(interpolation_interval[1] + .1, 6)
        )
        curr_func.change_gap(interpolation_interval)
    elif key == 'ctrl+right':
        interpolation_interval = (
            round(interpolation_interval[0] - .5, 6),
            round(interpolation_interval[1] + .5, 6)
        )
        curr_func.change_gap(interpolation_interval)
    elif key == 'shift+right':
        interpolation_interval = (
            round(interpolation_interval[0] - .05, 6),
            round(interpolation_interval[1] + .05, 6)
        )
        curr_func.change_gap(interpolation_interval)
    elif key == 'shift+ctrl+right':
        interpolation_interval = (
            round(interpolation_interval[0] - .01, 6),
            round(interpolation_interval[1] + .01, 6)
        )
        curr_func.change_gap(interpolation_interval)
    elif key == 'left':
        interpolation_interval = (
            min(round(interpolation_interval[0] + .1, 6), -.3),
            max(round(interpolation_interval[1] - .1, 6), .3)
        )
        curr_func.change_gap(interpolation_interval)
    elif key == 'ctrl+left':
        interpolation_interval = (
            min(round(interpolation_interval[0] + .5, 6), -.3),
            max(round(interpolation_interval[1] - .5, 6), .3)
        )
        curr_func.change_gap(interpolation_interval)
    elif key == 'shift+left':
        interpolation_interval = (
            min(round(interpolation_interval[0] + .05, 6), -.3),
            max(round(interpolation_interval[1] - .05, 6), .3)
        )
        curr_func.change_gap(interpolation_interval)
    elif key == 'shift+ctrl+left':
        interpolation_interval = (
            min(round(interpolation_interval[0] + .01, 6), -.3),
            max(round(interpolation_interval[1] - .01, 6), .3)
        )
        curr_func.change_gap(interpolation_interval)

    main()


if __name__ == "__main__":
    interpolation_interval = (-1., 1.)
    curr_number_of_nodes = 7
    curr_degree = 2
    curr_func = FunctionOnGap(FUNCS[1], interpolation_interval)

    fg, axis = plt.subplots(1, 2)
    fg.canvas.mpl_connect('key_release_event', on_key_event)

    main()
