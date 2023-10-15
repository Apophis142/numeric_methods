import math as m
import matplotlib.pyplot as plt
from lab2 import Matrix, solve_system
from random import random


def f_y(x):
    return 4*m.sin(x*2 + 1) - 1.2


def f_x(y):
    return -3*m.cos(y) / 2 + 1


def f_1(v: Matrix):
    x, y = v.t()[0]
    return 4*m.sin(x*2 + 1) - y - 1.2


def f_2(v: Matrix):
    x, y = v.t()[0]
    return 2*x + 3*m.cos(y) - 2


def f_1_dx(v: Matrix):
    return 8*m.cos(v[0][0]*2 + 1)


def f_1_dy(v: Matrix):
    return -1


def f_2_dx(v: Matrix):
    return 2


def f_2_dy(v: Matrix):
    return -3*m.sin(v[1][0])


# def f_y(x):
#     return m.sin(x + 1) + .8
#
#
# def f_x(y):
#     return 1.3 - m.sin(y - 1)
#
#
# def f_1(v: Matrix):
#     x, y = v.t()[0]
#     return y - m.sin(x + 1) - .8
#
#
# def f_2(v: Matrix):
#     x, y = v.t()[0]
#     return m.sin(y - 1) + x - 1.3
#
#
# def f_1_dx(v: Matrix):
#     return -m.cos(v[0][0] + 1)
#
#
# def f_1_dy(v: Matrix):
#     return 1
#
#
# def f_2_dx(v: Matrix):
#     return 1
#
#
# def f_2_dy(v: Matrix):
#     return m.cos(v[1][0] - 1)


def find_solution(init_val: Matrix, acc: float = 1e-4, counter: bool = False):
    prev_sol = init_val
    sol = prev_sol + solve_system(
        Matrix([[f_1_dx(prev_sol), f_1_dy(prev_sol)], [f_2_dx(prev_sol), f_2_dy(prev_sol)]]),
        Matrix([[-f_1(prev_sol), -f_2(prev_sol)]]).t()
    )
    step = 0
    g = None
    while abs(prev_sol-sol) > acc:
        if not g:
            g = Matrix([[f_1_dx(sol), f_1_dy(sol)], [f_2_dx(sol), f_2_dy(sol)]])
            # print(g)
        prev_sol, sol = sol, sol + solve_system(
            g,
            Matrix([[-f_1(sol), -f_2(sol)]]).t()
        )
        step += 1
        # print(prev_sol[0][0] - sol[0][0], prev_sol[1][0] - sol[1][0])
        if abs(g.det()) > 1e-6:
            g = None
        if step == 20:
            return Matrix([[6], [6]])
    # print(step)
    if counter:
        return step
    return sol


def axis():
    plt.plot([0, 0], [-5, 5], "--b", alpha=0.4)
    plt.plot([-5, 5], [0, 0], "--b", alpha=0.4)


def graphics():
    xs1, ys1 = [], []
    xs2, ys2 = [], []
    for arg in range(-500, 501):
        arg = arg / 100
        xs1 += [arg]
        ys2 += [arg]

        ys1 += [f_y(arg)]
        xs2 += [f_x(arg)]

    plt.subplot(111)
    plt.plot(xs1, ys1, "--", color="sienna", label="4sin(2x+1)-y=1.2", alpha=0.66)
    plt.plot(xs2, ys2, "--g", label="2x+3cos(y)=2", alpha=0.66)


def solutions():
    res = {(6, 6): []}
    for i in range(-80, 81):
        for j in range(-138, 81):
            init_val = Matrix([[i/16], [j/16]])
            solution = find_solution(init_val).to_tuple()
            flag = False
            for key in res:
                if abs(Matrix([key]) - Matrix([solution])) < 1e-4:
                    res[key] += [init_val]
                    flag = True
                    break
            if not flag:
                res[solution] = [init_val]

    show_solutions(res)


def show_solutions(sols: dict):
    colors = ["#ffffff", "#2ed500", "#3d14af", "#000000", "#c30083", "#fbfe00", "#04819e", "#cfd300"]
    i = 0
    for key in sols:
        if key != (6, 6):
            plt.plot([key[0]], [key[1]], "x", color=colors[i])
        print(key, len(sols[key]))
        plt.scatter([sol[0] for sol in sols[key]], [sol[1] for sol in sols[key]], color=colors[i],
                    alpha=(0.75 if key != (6, 6) else 0.25), s=0.2)
        i += 1


def rand_init_vals(sol: Matrix):
    init_vals_steps = []
    init_vals = []
    for _ in range(100):
        delta = Matrix([[(random() - 0.5)*2], [(random() - 0.5)*2]])
        init_val = sol + delta
        init_vals += [init_val]
        init_vals_steps += [[(delta[0][0]**2 + delta[1][0]**2)**0.5, find_solution(init_val, counter=True, acc=1e-8)]]

    plt.plot([*(init_vals[i][0][0] for i in range(100))], [*(init_vals[i][1][0] for i in range(100))], ".r", alpha=0.25,
             label="Начальные точки")

    plt.subplots(figsize=(10, 6))
    init_vals_steps.sort()
    c = Matrix(init_vals_steps).t()
    plt.plot(c[0], c[1], "-g")
    plt.xlabel("Расстояние до точки")
    plt.ylabel("Количество шагов для вычисления")


if __name__ == "__main__":
    plt.subplots(figsize=(8, 8))
    axis()
    # graphics()
    solutions()
    # plt.legend()

    # rand_init_vals(find_solution(Matrix([[0, 0]]).t()))
    plt.show()
