from polynomial import Polynomial
from lab2 import Matrix, solve_system as solve_slae
from function_on_gap import FunctionOnGap, FunctionSequence
from lagrange_interpolation import interpolation_polynomial as lip
from hermit_interpolation import interpolation_polynomial as hip
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
from interpolation_functions import INTERPOLATION_FUNCTIONS as FUNCS
from interpolation import ln_diff_functions


def linear_spline(table: list[tuple[float, float], ...]) -> FunctionSequence:
    res = []
    for j in range(len(table) - 1):
        pol = lip(table[j:j+2])
        res += [FunctionOnGap(pol, (table[j][0], table[j+1][0]))]

    return FunctionSequence(tuple(res))


def square_spline(table: list[tuple[float, float], ...], d0: float = 0) -> FunctionSequence:
    res = []
    pol = Polynomial([d0, 0])
    for j in range(len(table) - 1):
        pol = hip([(*table[j], pol.derivative()(table[j][0])), table[j+1]])
        res += [FunctionOnGap(pol, (table[j][0], table[j+1][0]))]

    return FunctionSequence(tuple(res))


def cube_spline(table: list[tuple[float, float], ...], first_point_second_der: float = 0,
                last_point_second_der: float = 0) -> FunctionSequence:
    n = len(table) - 1

    y_values = [point[1] for point in table]

    hs = [table[j+1][0] - table[j][0] for j in range(n)]
    h = Matrix([[hs[i] if i == j + 1 else
                 2*(hs[i] + hs[i+1]) if i == j else
                 hs[i+1] if i == j - 1 else 0
                 for j in range(n-1)] for i in range(n-1)])
    gamma = [[6 * ((y_values[i+1] - y_values[i]) / hs[i] - (y_values[i] - y_values[i-1]) / hs[i-1])]
                         for i in range(1, n)]
    gamma[0][0] -= first_point_second_der
    gamma[-1][0] -= last_point_second_der
    gamma = Matrix(gamma)

    y_second_der = [first_point_second_der] + list(solve_slae(h, gamma).t().matrix[0]) + [last_point_second_der]

    y_first_der = [(y_values[i+1] - y_values[i]) / hs[i] - y_second_der[i+1] * hs[i]/6 - y_second_der[i] * hs[i]/3
                   for i in range(n)]

    res = []
    for j in range(n):
        p = Polynomial([1, -table[j][0]])
        pol = Polynomial([y_values[j]]) + y_first_der[j] * p + 1/2 * y_second_der[j] * p**2 + \
              (y_second_der[j+1] - y_second_der[j]) / 6 / hs[j] * p**3
        res += [FunctionOnGap(pol, (table[j][0], table[j + 1][0]))]

    return FunctionSequence(tuple(res))


def show_linear_spline(table: list[tuple[float, float], ...], number_of_nodes: int) -> None:
    global fg, axis

    nodes = [[], []]
    for point in table:
        nodes[0] += [point[0]]
        nodes[1] += [point[1]]
    spline = linear_spline(table)

    fg.clear()
    ax = fg.add_subplot(axis[0:2, 0:2])
    for points in spline.points_of_func(number_of_nodes):
        ax.plot(*points, '-b')
    ax.plot([], [], '-b', label="сплайн")
    ax.plot(*curr_func.points_of_func(1000), 'r', alpha=.7, label="иск. функция")
    ax.legend()
    ax.set_title("Функция и линейный сплайн")

    ax.scatter(nodes[0], nodes[1], color='b')

    ax = fg.add_subplot(axis[0:2, 2])
    error = FunctionOnGap(ln_diff_functions(curr_func, spline), interpolation_interval)
    ax.plot(*error.points_of_func(1000), '--r')
    ax.set_title("Порядок погрешности")

    plt.show()


def show_square_spline(table: list[tuple[float, float], ...], number_of_nodes: int, d0: float = 0) -> None:
    global fg, axis

    nodes = [[], []]
    for point in table:
        nodes[0] += [point[0]]
        nodes[1] += [point[1]]
    spline = square_spline(table, d0)

    fg.clear()
    ax = fg.add_subplot(axis[0:2, 0])
    for points in spline.points_of_func(number_of_nodes):
        ax.plot(*points, '-b')
    ax.plot([], [], '-b', label="сплайн")
    ax.plot(*curr_func.points_of_func(1000), 'r', alpha=.7, label="иск. функция")
    ax.scatter(nodes[0], nodes[1], color='b')
    ax.legend()
    ax.set_title("Функция и квадратичный сплайн")

    ax = fg.add_subplot(axis[0:2, 1])
    for points in spline.derivative().points_of_func(number_of_nodes):
        ax.plot(*points, '-b')
    ax.plot(*curr_func.derivative().points_of_func(1000), 'r', alpha=.7)
    ax.set_title("Первая производная")

    ax = fg.add_subplot(axis[0:2, 2])
    error = FunctionOnGap(ln_diff_functions(curr_func, spline), interpolation_interval)
    ax.plot(*error.points_of_func(1000), '--r')
    ax.set_title("Порядок погрешности")

    plt.show()


def show_cube_spline(table: list[tuple[float, float], ...], number_of_nodes: int, first_point_second_der: float = 0,
                last_point_second_der: float = 0) -> None:
    global fg, axis

    nodes = [[], []]
    for point in table:
        nodes[0] += [point[0]]
        nodes[1] += [point[1]]
    spline = cube_spline(table, first_point_second_der, last_point_second_der)

    fg.clear()

    ax = fg.add_subplot(axis[0, 0:2])
    for points in spline.points_of_func(number_of_nodes):
        ax.plot(*points, '-b')
    ax.plot([], [], '-b', label="сплайн")
    ax.plot(*curr_func.points_of_func(1000), 'r', alpha=.7, label="иск. функция")
    ax.scatter(nodes[0], nodes[1], color='b', alpha=.7)
    ax.legend()
    ax.set_title("Функция и кубический сплайн")

    ax = fg.add_subplot(axis[1, 0])
    for points in spline.derivative().points_of_func(number_of_nodes):
        ax.plot(*points, '-b')
    ax.plot(*curr_func.derivative().points_of_func(1000), 'r', alpha=.7)
    ax.set_title("Первая производная")

    ax = fg.add_subplot(axis[1, 1])
    for points in spline.derivative(2).points_of_func(number_of_nodes):
        ax.plot(*points, '-b')
    ax.plot(*curr_func.derivative(2).points_of_func(1000), 'r', alpha=.7)
    ax.set_title("Вторая производная")

    ax = fg.add_subplot(axis[0:2, 2])
    error = FunctionOnGap(ln_diff_functions(curr_func, spline), interpolation_interval)
    ax.plot(*error.points_of_func(1000), '--r')
    ax.set_title("Порядок погрешности")

    plt.show()


def on_key_event(event) -> None:
    global curr_number_of_nodes
    global interpolation_interval
    global curr_func, curr_spline
    key = getattr(event, 'key')

    if key == 'escape':
        interpolation_interval = (-1., 1.)
        curr_spline = 3
        curr_number_of_nodes = 5
        curr_func = FunctionOnGap(FUNCS[1], interpolation_interval)
    elif key == 'ctrl+l':
        curr_spline = 1
    elif key == 'ctrl+q':
        curr_spline = 2
    elif key == 'ctrl+c':
        curr_spline = 3
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
    elif key in ('ctrl+plus', 'ctrl+eq', 'ctrl+up'):
        curr_number_of_nodes += 5
    elif key in ('ctrl+minus', 'ctrl+down'):
        curr_number_of_nodes = max(3, curr_number_of_nodes - 5)
    elif key == 'right':
        interpolation_interval = (
            round(interpolation_interval[0] - .1, 6),
            round(interpolation_interval[1] + .1, 6)
        )
    elif key == 'ctrl+right':
        interpolation_interval = (
            round(interpolation_interval[0] - .5, 6),
            round(interpolation_interval[1] + .5, 6)
        )
    elif key == 'shift+right':
        interpolation_interval = (
            round(interpolation_interval[0] - .05, 6),
            round(interpolation_interval[1] + .05, 6)
        )
    elif key == 'shift+ctrl+right':
        interpolation_interval = (
            round(interpolation_interval[0] - .01, 6),
            round(interpolation_interval[1] + .01, 6)
        )
    elif key == 'left':
        interpolation_interval = (
            min(round(interpolation_interval[0] + .1, 6), -.3),
            max(round(interpolation_interval[1] - .1, 6), .3)
        )
    elif key == 'ctrl+left':
        interpolation_interval = (
            min(round(interpolation_interval[0] + .5, 6), -.3),
            max(round(interpolation_interval[1] - .5, 6), .3)
        )
    elif key == 'shift+left':
        interpolation_interval = (
            min(round(interpolation_interval[0] + .05, 6), -.3),
            max(round(interpolation_interval[1] - .05, 6), .3)
        )
    elif key == 'shift+ctrl+left':
        interpolation_interval = (
            min(round(interpolation_interval[0] + .01, 6), -.3),
            max(round(interpolation_interval[1] - .01, 6), .3)
        )

    main()


def main() -> None:
    table = curr_func.points_of_func(curr_number_of_nodes)
    table = [(table[0][j], table[1][j]) for j in range(len(table[0]))]

    if curr_spline == 1:
        show_linear_spline(table, 1000)
    elif curr_spline == 2:
        show_square_spline(table, 1000, curr_func.derivative()(interpolation_interval[0]))
    elif curr_spline == 3:
        show_cube_spline(table, 1000)

    plt.show()


if __name__ == "__main__":
    interpolation_interval = (-1., 1.)
    curr_spline = 3
    curr_number_of_nodes = 5
    curr_func = FunctionOnGap(FUNCS[1], interpolation_interval)

    fg = plt.figure()
    axis = gsp.GridSpec(ncols=3, nrows=2, figure=fg)
    fg.canvas.mpl_connect('key_release_event', on_key_event)
    main()
