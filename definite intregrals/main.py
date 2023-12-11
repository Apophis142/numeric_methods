from scipy.integrate import quad
import matplotlib.pyplot as plt
from math import log10, sin, exp
from quadrature_formulas import *
from error_estimate import *


true_value: tuple


def calculate_method(method: callable([Function, float, float])) -> tuple[list[int, ...], list[float, ...]]:
    l1, l2 = [], []
    for n in range(1, 101):
        l1 += [n]
        l2 += [integrate(func, GL_a, GL_b, n, method)]

    return l1, l2


def check_method_error(method: callable([Function, float, float])) -> tuple[list[int, ...], list[float, ...]]:
    l1, l2 = calculate_method(method)
    l2 = [log10(abs(x - true_value[0])) for x in l2]
    return l1, l2


def show_method_error(method: callable([Function, float, float]),
                 color:str, label: str, alpha: float = .75, linewidth: float=.5) -> None:
    plt.subplot(1, 2, 1)
    axis[0].plot(*check_method_error(method), color, alpha=alpha, label=label, linewidth=linewidth)


def show_base_error() -> None:
    plt.subplot(1, 2, 1)
    plt.plot((1, 101), [log10(true_value[1])] * 2, '-r', linewidth=.75, alpha=.75,
                 label="погрешность истинного значения")


def show_graphics1() -> None:
    global true_value
    true_value = quad(func.func, GL_a, GL_b)

    show_base_error()
    show_method_error(left_rectangle_formula, color='--g',
                      label="формула левых прямоугольников", alpha=1)
    show_method_error(right_rectangle_formula, color='--b',
                      label="формула правых прямоугольников", alpha=1)
    show_method_error(middle_rectangle_formula, color='--y',
                      label="формула средних прямоугольников", alpha=1)
    show_method_error(trapezoid_formula, color='--k',
                      label="формула трапеции", alpha=1)
    show_method_error(simpson_formula, color='--c',
                      label="формула Симпсона", alpha=1)

    plt.legend()
    plt.xlabel("Количество отрезков разбиения")
    plt.ylabel("Порядок погрешности")
    plt.show()


def show_graphics2(method: callable([Function, float, float])) -> None:
    global true_value
    true_value = quad(lambda x: func(x) / ((x - GL_a)**GL_alpha * (GL_b - x)**GL_beta), GL_a, GL_b)
    max_r = 5

    plt.subplot(1, 2, 1)
    n_parts, integral_vals = calculate_method(method)
    better_integral_vals, integral_value_error = [], []

    error = 1
    i = 2
    while error > 1e-6:
        m = aitken(func, method, GL_a, GL_b, n_parts[i])
        r = min(max_r, i)
        val, error = richardson(m, n_parts[i - r : i], integral_vals[i - r : i], GL_a, GL_b)
        better_integral_vals += [val]
        integral_value_error += [error]

        i += 1
        print(i)

    plt.plot(n_parts, list(  map(lambda x: log10(  abs(x - true_value[0])  ), integral_vals)  ),
             '-g', alpha=.75, linewidth=.5, label="погр. от ист. знач.")
    plt.plot(n_parts[:i-2], list(  map(lambda x: log10(  abs(x - true_value[0])  ), better_integral_vals)  ),
             '--r', alpha=.75, linewidth=.5, label="погр. улучш. знач.")
    plt.plot(n_parts[:i-2], list(  map(log10, integral_value_error)  ),
             '-.k', alpha=.75, linewidth=.5, label="расч. погр.")
    show_base_error()

    plt.legend()
    plt.xlabel("Количество отрезков разбиения")
    plt.ylabel("Порядок погрешности")

    aitken_show(method)

    plt.show()


def aitken_show(method: callable([Function, float, float])):
    plt.subplot(1, 2, 2)
    axis[1].plot(range(1, 26), [aitken(func, method, GL_a, GL_b, x) for x in range(1, 26)])
    plt.xlabel("Сетка разбиения")
    plt.xticks([1, 5, 10, 15, 20, 25], ["{1, 2, 4}", "{5, 10, 20}", "{10, 20, 40}",
                                        "{15, 30, 60}", "{20, 40, 80}", "{25, 50, 100}"])
    plt.ylabel("Расчетный порядок гл. члена погр.")


if __name__ == "__main__":
    func = Function(lambda x: 2 * cos(2.5 * x) * exp(x / 3) + 4 * sin(3.5 * x) * exp(-3 * x) + x)

    fg, axis = plt.subplots(1, 2)
    # show_graphics1()
    show_graphics2(newton_kotes_formula)
