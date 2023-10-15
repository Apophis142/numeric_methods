import math as m
from functools import cache
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp


class Pair(object):
    val: float
    terms_number: int

    def __init__(self, val, terms_number):
        self.val = val
        self.terms_number = terms_number

    def __add__(self, other):
        if isinstance(other, Pair):
            return Pair(self.val + other.val, max(self.terms_number, other.terms_number))
        else:
            raise ValueError

    def __neg__(self):
        return Pair(-self.val, self.terms_number)

    def __sub__(self, other):
        return self + (-other)


def float_range(begin: float, end: float, step: float) -> list[float]:
    return [begin + x * step for x in range(int((end - begin) // step) + 1)]


def sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


@cache
def my_pow(arg: float, power: int) -> float:
    if arg == 1:
        return arg
    elif arg == -1:
        return (power % 2) * (-2) + 1
    if power == 0:
        return 1.0
    elif power < 0:
        return 1 / my_pow(arg, -power)
    elif power == 1:
        return arg
    elif power == 2:
        return arg * arg
    elif power % 2:
        return my_pow(my_pow(arg, power // 2), 2) * arg
    elif not (power % 2):
        return my_pow(my_pow(arg, power // 2), 2)


def my_sqrt(arg: float, acc: float = 1e-6) -> float:
    if arg == 0:
        return 0
    w, prev_w = .5 * (max(1., arg) + arg / max(1., arg)), -m.inf
    while abs(w - prev_w) >= acc:
        prev_w, w = w, .5 * (w + arg / w)
    return w


def taylor_series(common_term, init_val: float = 0, acc: float = 1e-6) -> Pair:
    res = init_val
    k = 0
    while abs(x := common_term(k)) > acc:
        res += x
        k += 1
    return Pair(res, k - 1)


def epx_common_term(arg: float):
    return lambda k: my_pow(arg, k) / m.factorial(k)


def sh_common_term(arg: float):
    return lambda k: my_pow(arg, 2 * k + 1) / m.factorial(2 * k + 1)


def ch_common_term(arg: float):
    return lambda k: my_pow(arg, 2 * k) / m.factorial(2 * k)


def sin_common_term(arg: float):
    return lambda k: my_pow(-1, k) * my_pow(arg, 2 * k + 1) / m.factorial(2 * k + 1)


def cos_common_term(arg: float):
    return lambda k: my_pow(-1, k) * my_pow(arg, 2 * k) / m.factorial(2 * k)


def atg_init_val(arg: float) -> float:
    return m.pi / 2 * sign(arg) if abs(arg) >= 1 else 0


def atg_common_term(arg: float):
    if abs(arg) < 1:
        return lambda k: my_pow(-1, k) * my_pow(arg, 2 * k + 1) / (2 * k + 1)
    else:
        return lambda k: my_pow(-1, k + 1) * my_pow(arg, -(2 * k + 1)) / (2 * k + 1)


def my_func_u(arg: float, acc: float = 1e-6) -> Pair:
    return Pair(1., 0) + taylor_series(atg_common_term(16.7 * arg + .1), atg_init_val(16.7 * arg + .1), acc=acc)


def my_func_v(arg: float, acc: float = 1e-6) -> Pair:
    return taylor_series(cos_common_term(7 * arg + .3), acc=acc)


def my_func_f(val_u: float, val_v: float, acc: float = 1e-6) -> float:
    return my_sqrt(val_u, acc=acc) / val_v


def func_u(arg: float) -> float:
    return 1 + m.atan(16.7 * arg + .1)


def func_v(arg: float) -> float:
    return m.cos(7 * arg + .3)


def func_f(val_v: float, val_u: float) -> float:
    return val_v ** .5 / val_u


def main(gap_begin: float, gap_end: float, gap_step: float, epsilon, me=1) -> None:
    # 1 — принцип равных влияний; 2 — принцип равных погрешностей

    def f_u(*args):
        _u, _v = args
        return 1 / (2 * m.sqrt(_u) * _v)

    def f_v(*args):
        _u, _v = args
        return -m.sqrt(_u) / (_v ** 2)

    gap = float_range(gap_begin, gap_end, gap_step)

    error = epsilon
    res_error_u1, res_error_v1, res_error_f1 = [], [], []
    res_error_u2 = res_error_v2 = []

    res_f, res_my_f1, res_f2, res_my_f2 = [], [], [], []
    res_u, res_v = [], []
    res_my_u2, res_my_v2, res_my_u1, res_my_v1 = [], [], [], []
    res_acc_u1, res_acc_v1, res_acc1 = [], [], []
    res_acc_u2, res_acc_v2, res_acc2 = [], [], []
    res_lg_acc_u1, res_lg_acc_v1, res_lg_acc1 = [], [], []
    res_lg_acc_u2, res_lg_acc_v2, res_lg_acc2 = [], [], []

    atan_num1, atan_num2 = [], []
    cos_num1, cos_num2 = [], []

    lg = m.log10
    for arg in gap:
        b_u = abs(f_u(func_u(arg), func_v(arg)))
        b_v = abs(f_v(func_u(arg), func_v(arg)))
        res_error_u1 += [error_u1 := error / (3 * b_u)]
        res_error_v1 += [error_v1 := error / (3 * b_v)]
        error_f1 = error / 3
        res_error_u2 += [error_u2 := error / (1 + b_u + b_v)]

        res_u += [u := func_u(arg)]
        res_v += [v := func_v(arg)]
        res_f += [f := func_f(u, v)]
        res_my_u1 += [(my_u1 := my_func_u(arg, acc=error_u1)).val]
        res_my_v1 += [(my_v1 := my_func_v(arg, acc=error_v1)).val]
        res_my_f1 += [my_f1 := my_func_f(my_u1.val, my_v1.val, acc=error_f1)]
        atan_num1 += [my_u1.terms_number]
        cos_num1 += [my_v1.terms_number]
        res_my_u2 += [(my_u2 := my_func_u(arg, acc=error_u2)).val]
        res_my_v2 += [(my_v2 := my_func_v(arg, acc=error_u2)).val]
        res_my_f2 += [my_f2 := my_func_f(my_u2.val, my_v2.val, acc=error_u1)]
        atan_num2 += [my_u2.terms_number]
        cos_num2 += [my_v2.terms_number]
        res_acc_u1 += [du1 := abs(u - my_u1.val)]
        res_acc_v1 += [dv1 := abs(v - my_v1.val)]
        res_acc1 += [df1 := abs(f - my_f1)]
        res_acc_u2 += [du2 := abs(u - my_u2.val)]
        res_acc_v2 += [dv2 := abs(v - my_v2.val)]
        res_acc2 += [df2 := abs(f - my_f2)]
        res_lg_acc_u1 += [lg(du1) if du1 else -16]
        res_lg_acc_v1 += [lg(dv1) if dv1 else -16]
        res_lg_acc1 += [lg(df1) if df1 else -16]
        res_lg_acc_u2 += [lg(du2) if du2 else -16]
        res_lg_acc_v2 += [lg(dv2) if dv2 else -16]
        res_lg_acc2 += [lg(df2) if df2 else -16]

    show_graphics_f(gap, res_f, res_my_f1, res_my_f2, res_acc1, res_acc2, res_lg_acc1, res_lg_acc2, mark_every=me)
    show_graphics_ts(gap, res_u, res_my_u1, res_my_u2, res_acc_u1, res_acc_u2, res_lg_acc_u1, res_lg_acc_u2,
                     atan_num1, atan_num2, res_error_u1, res_error_u2, mark_every=me)
    show_graphics_ts(gap, res_v, res_my_v1, res_my_v2, res_acc_v1, res_acc_v2, res_lg_acc_v1, res_lg_acc_v2,
                     cos_num1, cos_num2, res_error_v1, res_error_v2, mark_every=me)

    plt.show()


def show_graphics_f(
        args: list[float],
        res: list[float],
        my_res_equal_influences: list[float], my_res_equal_errors: list[float],
        acc_equal_influences: list[float], acc_equal_errors: list[float],
        lg_acc_equal_influences: list[float], lg_acc_equal_errors: list[float],
        mark_every: int = 1
) -> None:
    fg = plt.figure()
    gs = gsp.GridSpec(ncols=2, nrows=2, figure=fg)
    fig_ax_1 = fg.add_subplot(gs[0, 0])
    plt.plot(args, res, "-r", label="Эталон", markevery=mark_every)
    fig_ax_1.set_title("Эталон")

    fig_ax_2 = fg.add_subplot(gs[0, 1])
    plt.plot(args, my_res_equal_influences, "-.c", label="равн. влиян.", markevery=mark_every)
    plt.plot(args, my_res_equal_errors, "--r", label="равн. погр.", markevery=mark_every)
    fig_ax_2.set_title("Приближенное значение")
    fig_ax_2.legend()

    fig_ax_3 = fg.add_subplot(gs[1, 0])
    plt.plot(args, acc_equal_influences, ".-g", label="равн. влиян.", alpha=.5, markevery=mark_every)
    plt.plot(args, acc_equal_errors, ".-b", label="равн. погр.", alpha=.5, markevery=mark_every)
    fig_ax_3.set_title("Абсолютная погрешность")
    fig_ax_3.legend()

    fig_ax_4 = fg.add_subplot(gs[1, 1])
    plt.plot(args, lg_acc_equal_influences, ".-g", label="равн. влиян.", alpha=.5, markevery=mark_every)
    plt.plot(args, lg_acc_equal_errors, ".-b", label="равн. погр.", alpha=.5, markevery=mark_every)
    fig_ax_4.set_title("Порядок погрешности")
    fig_ax_4.legend()


def show_graphics_ts(
        args: list[float],
        res: list[float],
        my_res_equal_influences: list[float], my_res_equal_errors: list[float],
        acc_equal_influences: list[float], acc_equal_errors: list[float],
        lg_acc_equal_influences: list[float], lg_acc_equal_errors: list[float],
        terms_number1: list[int], terms_number2: list[int],
        b1_equal_influences: list[float], b2_equal_errors: list[float],
        mark_every: int = 1
) -> None:
    fg = plt.figure()
    gs = gsp.GridSpec(ncols=3, nrows=2, figure=fg)

    fig_ax_1 = fg.add_subplot(gs[0, 0])
    plt.plot(args, res, "-r", label="Эталон", markevery=mark_every)
    fig_ax_1.set_title("Эталон")

    fig_ax_2 = fg.add_subplot(gs[0, 1])
    plt.plot(args, my_res_equal_influences, "-.c", label="равн. влиян.", alpha=.5, markevery=mark_every)
    plt.plot(args, my_res_equal_errors, "--r", label="равн. погр.", alpha=.5, markevery=mark_every)
    fig_ax_2.set_title("Приближенное значение")
    fig_ax_2.legend()

    fig_ax_3 = fg.add_subplot(gs[1, 0])
    plt.plot(args, acc_equal_influences, ".-g", label="равн. влиян.", alpha=.5, markevery=mark_every)
    plt.plot(args, acc_equal_errors, ".-b", label="равн. погр.", alpha=.5, markevery=mark_every)
    fig_ax_3.set_title("Абсолютная погрешность")
    fig_ax_3.legend()

    fig_ax_4 = fg.add_subplot(gs[1, 1])
    plt.plot(args, lg_acc_equal_influences, ".-g", label="равн. влиян.", alpha=.5, markevery=mark_every)
    plt.plot(args, lg_acc_equal_errors, ".-b", label="равн. погр.", alpha=.5, markevery=mark_every)
    fig_ax_4.set_title("Порядок погрешности")
    fig_ax_4.legend()

    fig_ax_5 = fg.add_subplot(gs[0, 2])
    plt.plot(args, terms_number1, ".-.g", label="равн. влиян.", alpha=.5, markevery=mark_every)
    plt.plot(args, terms_number2, ".-.c", label="равн. погр.", alpha=.5, markevery=mark_every)
    fig_ax_5.set_title("Количество членов при разложении")
    fig_ax_5.legend()

    fig_ax_6 = fg.add_subplot(gs[1, 2])
    plt.plot(args, b1_equal_influences, ".-.g", label="равн. влиян.", alpha=.5, markevery=mark_every)
    plt.plot(args, b2_equal_errors, ".-.c", label="равн. погр.", alpha=.5, markevery=mark_every)
    fig_ax_6.set_title("Расчетные погрешности")
    fig_ax_6.legend()


if __name__ == "__main__":
    main(.01, .05, .005, 1e-10)
