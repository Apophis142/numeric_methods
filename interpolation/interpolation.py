import math as m
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from polynomial import Polynomial
from lagrange_interpolation import interpolation_polynomial as lip
from newton_interpolation import sequence_interpolation_polynomials as sequence_nip
from interpolation_functions import INTERPOLATION_FUNCTIONS as FUNCS


def get_node_polynomial(nodes: list) -> Polynomial:
    res = Polynomial([1])
    for x in nodes:
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[0]
        res *= Polynomial([1, -x])
    return res


def points_of_func(function: callable(float), method: callable((int, float, float)), begin: float, end: float, n: int) \
        -> list[tuple[float, float], ...]:
    res = []
    for x in method(n, begin, end):
        res += [(x, function(x))]
    return res


def paint_func(function: callable(float), begin: float, end: float, n: int) \
        -> tuple[list[float, ...], ...]:
    res = points_of_func(function, choose_equidistant_interpolation_nodes, begin, end, n)
    xs, ys = [], []
    for point in res:
        xs += [point[0]]
        ys += [point[1]]
    return xs, ys


def choose_equidistant_interpolation_nodes(n: int, begin: float, end: float) -> list[float, ...]:
    def float_range(_begin: float, _end: float, _step: float) -> list[float, ...]:
        return [_begin + x * _step for x in range(int((_end - _begin) // _step) + 1)]

    step = (end - begin) / n
    return float_range(begin, end + step / 2, step)


def choose_chebyshev_interpolation_nodes(n: int, begin: float, end: float) -> list[float, ...]:
    res = []
    for i in range(0, n + 1):
        res += [1/2 * ((end - begin) * m.cos((2*i + 1) / (2*(n + 1)) * m.pi) + (begin + end))]
    return sorted(res)


def ln_diff_functions(_func1: callable(float), _func2: callable(float)) -> callable(float):
    return lambda x: m.log10(y) if (y := abs(_func1(x) - _func2(x))) else -16


def min_max(arr: (tuple[float, ...], list[float, ...])) -> tuple[float, float]:
    curr_min = curr_max = arr[1][0]
    for y in arr[1]:
        if curr_max < y:
            curr_max = y
        elif curr_min > y:
            curr_min = y
    return curr_min, curr_max


def error_estimate(node_polynomial: Polynomial) -> float:
    node_polynomial_error = max(abs(x) for x in min_max(paint_func(node_polynomial, *interpolation_interval, 1000)))
    if node_polynomial.degree >= 2:
        return node_polynomial_error / m.factorial(node_polynomial.degree)
    elif node_polynomial.degree == 1:
        return node_polynomial_error


def show_lagrange_graphics(i: int) -> None:
    i -= 1
    interval = (*interpolation_interval, i)
    interpolation_nodes1 = points_of_func(curr_func, choose_equidistant_interpolation_nodes, *interval)
    interpolation_nodes2 = points_of_func(curr_func, choose_chebyshev_interpolation_nodes, *interval)
    interpolation1 = lip(interpolation_nodes1)
    interpolation2 = lip(interpolation_nodes2)

    interval_show = (interpolation_interval[0] * 1., interpolation_interval[1] * 1.,
                     int(650 * (interpolation_interval[1] - interpolation_interval[0])**.25))

    ax = axis[0]
    ax.clear()

    if not show_node_polynomial:
        ax.plot(*(func_points := paint_func(curr_func, *interval_show)),
                "-g", linewidth=0.5, alpha=.85, label="иск. функция")
        interpolation_borders = min_max(func_points)
        ax.plot([interpolation_interval[0]] * 2, interpolation_borders,
                "-y", alpha=.3,
                label=f"Границы интерполяции ({interpolation_interval[0]}; {interpolation_interval[1]})")
        ax.plot([interpolation_interval[1]] * 2, interpolation_borders,
                "-y", alpha=.3)

    if show_equidistant:
        if show_node_polynomial:
            node_polynomial = get_node_polynomial(
                [interpolation_nodes1[j][0] for j in range(i + 1)]
            )
            ax.plot(*paint_func(node_polynomial, *interval_show),
                    "-r", alpha=.5, label=f"узл. многочлен (по {i + 1} равноуд. т.)")
            ax.scatter(
                [p[0] for p in interpolation_nodes1], [0] * (i + 1),
                color='r', alpha=.5, s=7.5, marker='o'
            )
        else:
            ax.plot(*paint_func(interpolation1, *interval_show),
                    "--r", alpha=.6, label=f"интерп. полином (по {i + 1} равноуд. т.)")
            ax.scatter(
                [p[0] for p in interpolation_nodes1], [p[1] for p in interpolation_nodes1],
                color='r', alpha=.5, s=7.5, marker='o'
            )

    if show_chebyshev:
        if show_node_polynomial:
            node_polynomial = get_node_polynomial(
                [interpolation_nodes2[j][0] for j in range(i + 1)]
            )
            ax.plot(*paint_func(node_polynomial, *interval_show),
                    "-c", alpha=.5, label=f"узл. многочлен (по {i + 1} \"Чебышевским\". т.)")
            ax.scatter(
                [p[0] for p in interpolation_nodes2], [0] * (i+1),
                color='c', alpha=.5, s=7.5, marker='o'
            )
        else:
            ax.plot(*paint_func(interpolation2, *interval_show),
                    "--c", alpha=.5, label=f"интерп. полином (по {i + 1} \"Чебышевским\" т.)")
            ax.scatter(
                [p[0] for p in interpolation_nodes2], [p[1] for p in interpolation_nodes2],
                color='c', alpha=.5, s=7.5, marker='o'
            )

    ax.set_title("Функция и интерп. полином Лагранжа")
    ax.legend()

    ax = axis[1]
    ax.clear()
    if show_equidistant:
        ax.plot(*(x := paint_func(ln_diff_functions(curr_func, interpolation1), *interval_show)),
                "--r", alpha=.6, label=f"равноуд. т.")
        ax.plot(interpolation_interval, [m.log10(error_estimate(get_node_polynomial(interpolation_nodes1)))] * 2,
                "-.r", alpha=.6, label="оценка погр. для равноуд. т.")
    else:
        x = ((0, 0), (-16, 0))
    if show_chebyshev:
        ax.plot(*(y := paint_func(ln_diff_functions(curr_func, interpolation2), *interval_show)),
                "--b", alpha=.6, label=f"\"Чебышевские\". т.")
        ax.plot(interpolation_interval, [m.log10(error_estimate(get_node_polynomial(interpolation_nodes2)))] * 2,
                "-.b", alpha=.6, label="оценка погр. для \"Чебышевских\". т.")
    else:
        y = ((0, 0), (-16, 0))

    interpolation_borders = min_max(x + y)
    ax.plot([interpolation_interval[0]] * 2, interpolation_borders,
            "-y", alpha=.3)
    ax.plot([interpolation_interval[1]] * 2, interpolation_borders,
            "-y", alpha=.3)
    ax.set_title("Логарифм погрешности интерп. полиномов")
    plt.show()


def show_animation(name: str = 'interpolation') -> None:
    ani = animation.FuncAnimation(fg, show_lagrange_graphics, interval=1000, frames=9)
    if name:
        ani.save(name + '.gif')


def on_key_event(event) -> None:
    global curr_number_of_nodes
    global interpolation_interval
    global show_equidistant, show_chebyshev, show_node_polynomial
    global curr_func

    key = getattr(event, 'key')

    if key == 'escape':
        curr_number_of_nodes = 4
        # interpolation_interval = (-1., 1.)
        interpolation_interval = (0., 1.)
        show_equidistant = show_chebyshev = True
        show_node_polynomial = False
        curr_func = FUNCS[1]
    elif key in ('z', 'Z'):
        show_equidistant = not show_equidistant
    elif key in ('x', 'X'):
        show_chebyshev = not show_chebyshev
    elif key in ('n', 'N'):
        show_node_polynomial = not show_node_polynomial
    elif key.isdigit():
        curr_func = FUNCS[int(key)]
    elif key.startswith('ctrl+') and key[-1].isdigit():
        def tmp(x):
            return abs(x) * FUNCS[int(key[-1])](x)
        curr_func = tmp
    elif key in ('+', '=', 'up'):
        curr_number_of_nodes += 1
    elif key in ('-', 'down'):
        curr_number_of_nodes = max(1, curr_number_of_nodes - 1)
    elif key in ('ctrl+plus', 'ctrl+eq', 'ctrl+up'):
        curr_number_of_nodes += 5
    elif key in ('ctrl+minus', 'ctrl+down'):
        curr_number_of_nodes = max(1, curr_number_of_nodes - 5)
    elif key == 'right':
        interpolation_interval = (
            # round(interpolation_interval[0] - .1, 6),
            interpolation_interval[0],
            round(interpolation_interval[1] + .1, 6)
        )
    elif key == 'ctrl+right':
        interpolation_interval = (
            # round(interpolation_interval[0] - .5, 6),
            interpolation_interval[0],
            round(interpolation_interval[1] + .5, 6)
        )
    elif key == 'shift+right':
        interpolation_interval = (
            # round(interpolation_interval[0] - .05, 6),
            interpolation_interval[0],
            round(interpolation_interval[1] + .05, 6)
        )
    elif key == 'shift+ctrl+right':
        interpolation_interval = (
            # round(interpolation_interval[0] - .01, 6),
            interpolation_interval[0],
            round(interpolation_interval[1] + .01, 6)
        )
    elif key == 'left':
        interpolation_interval = (
            # min(round(interpolation_interval[0] + .1, 6), -.3),
            interpolation_interval[0],
            max(round(interpolation_interval[1] - .1, 6), .3)
        )
    elif key == 'ctrl+left':
        interpolation_interval = (
            # min(round(interpolation_interval[0] + .5, 6), -.3),
            interpolation_interval[0],
            max(round(interpolation_interval[1] - .5, 6), .3)
        )
    elif key == 'shift+left':
        interpolation_interval = (
            # min(round(interpolation_interval[0] + .05, 6), -.3),
            interpolation_interval[0],
            max(round(interpolation_interval[1] - .05, 6), .3)
        )
    elif key == 'shift+ctrl+left':
        interpolation_interval = (
            # min(round(interpolation_interval[0] + .01, 6), -.3),
            interpolation_interval[0],
            max(round(interpolation_interval[1] - .01, 6), .3)
        )

    show_lagrange_graphics(curr_number_of_nodes)
    plt.show()


def lagrange_main() -> None:
    fg.canvas.mpl_connect('key_release_event', on_key_event)
    show_lagrange_graphics(curr_number_of_nodes)
    plt.show()


def show_newton_graphics(iteration_number: int) -> None:
    iteration_number += 1

    interval = (*interpolation_interval, curr_number_of_nodes)
    interpolation_nodes1 = points_of_func(curr_func, choose_equidistant_interpolation_nodes, *interval)
    interpolation_nodes2 = points_of_func(curr_func, choose_chebyshev_interpolation_nodes, *interval)

    interval_show = (interpolation_interval[0] * 1., interpolation_interval[1] * 1.,
                     int(650 * (interpolation_interval[1] - interpolation_interval[0]) ** .25))
    interpolation1 = newt_pol_seq_eq[iteration_number]
    interpolation2 = newt_pol_seq_ch[iteration_number]

    ax = axis[0]
    ax.clear()

    ax.plot(*paint_func(curr_func, *interval_show), "-g", linewidth=0.5, alpha=.85, label="иск. функция")

    if show_equidistant:
        ax.plot(*paint_func(interpolation1, *interval_show),
                "--r", alpha=.6, label=f"интерп. полином (по {iteration_number + 1} равноуд. т.)")
        active_nodes1 = interpolation_nodes1[:iteration_number]
        inactive_nodes1 = interpolation_nodes1[iteration_number:]
        ax.scatter(
            [p[0] for p in active_nodes1], [p[1] for p in active_nodes1],
            color='r', alpha=.5, s=7.5, marker='o'
        )
        ax.scatter(
            [p[0] for p in inactive_nodes1], [p[1] for p in inactive_nodes1],
            color='r', alpha=.5, s=7.5, marker='x'
        )

    if show_chebyshev:
        ax.plot(*paint_func(interpolation2, *interval_show),
                "--c", alpha=.6, label=f"интерп. полином (по {iteration_number + 1} \"Чебышевским\" т.)")
        active_nodes2 = interpolation_nodes2[:iteration_number]
        inactive_nodes2 = interpolation_nodes2[iteration_number:]
        ax.scatter(
            [p[0] for p in active_nodes2], [p[1] for p in active_nodes2],
            color='c', alpha=.5, s=7.5, marker='o'
        )
        ax.scatter(
            [p[0] for p in inactive_nodes2], [p[1] for p in inactive_nodes2],
            color='c', alpha=.5, s=7.5, marker='x'
        )

    ax.set_title("Функция и интерп. полином Ньютона")
    ax.legend()

    ax = axis[1]
    ax.clear()
    if show_equidistant:
        ax.plot(*(x := paint_func(ln_diff_functions(curr_func, interpolation1), *interval_show)),
                "--r", alpha=.6, label=f"равноуд. т.")
    else:
        x = ((0, 0), (-16, 0))
    if show_chebyshev:
        ax.plot(*(y := paint_func(ln_diff_functions(curr_func, interpolation2), *interval_show)),
                "--c", alpha=.6, label=f"\"Чебышевские\". т.")
    else:
        y = ((0, 0), (-16, 0))

    interpolation_borders = min_max(x + y)
    ax.plot([interpolation_interval[0]] * 2, interpolation_borders,
            "-y", alpha=.3)
    ax.plot([interpolation_interval[1]] * 2, interpolation_borders,
            "-y", alpha=.3)
    ax.set_title("Логарифм погрешности интерп. полиномов")

    plt.show()


def newton_main(name: str = None) -> None:
    global newt_pol_seq_eq, newt_pol_seq_ch

    interval = (*interpolation_interval, curr_number_of_nodes)
    interpolation_nodes1 = points_of_func(curr_func, choose_equidistant_interpolation_nodes, *interval)
    interpolation_nodes2 = points_of_func(curr_func, choose_chebyshev_interpolation_nodes, *interval)
    # shuffle(interpolation_nodes1)
    # shuffle(interpolation_nodes2)

    newt_pol_seq_eq = sequence_nip(interpolation_nodes1)
    newt_pol_seq_ch = sequence_nip(interpolation_nodes2)

    ani = animation.FuncAnimation(fg, show_newton_graphics, interval=10000, frames=curr_number_of_nodes - 1)
    plt.show()
    if name:
        ani.save(name + '.gif')


if __name__ == "__main__":
    curr_number_of_nodes = 4
    interpolation_interval = (0., 1.)
    show_equidistant, show_chebyshev = True, True
    show_node_polynomial = False
    curr_func = FUNCS[1]

    newt_pol_seq_eq, newt_pol_seq_ch = [], []

    fg, axis = plt.subplots(1, 1)

    # fg.canvas.mpl_connect('key_release_event', on_key_event)
    # show_lagrange_graphics(curr_number_of_nodes)
    # plt.show()
