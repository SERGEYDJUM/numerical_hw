import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from math import factorial

Matrix = np.ndarray[(int, int), float]
Vector = np.ndarray[(int), float]


def diff_coeffs(Y: Vector, order: int) -> Vector:
    n = Y.shape[0]
    dif = np.zeros((order + 1, n), dtype=float)
    dif[0] = Y
    for ord in range(1, order + 1):
        dif[ord] = np.concatenate(
            (np.diff(dif[ord - 1][: (n - ord + 1)]), np.zeros(ord))
        )
    return dif[:, 0]


def divdiff_coeffs(XY: Matrix, order: int) -> Vector:
    n = XY.shape[0]
    dif = np.zeros((order + 1, n))
    dif[0] = XY[:, 1]
    for ord in range(1, order + 1):
        for i in range(n - ord):
            dif[ord, i] = (dif[ord - 1, i + 1] - dif[ord - 1, i]) / (
                XY[i + ord, 0] - XY[i, 0]
            )
    return dif[:, 0]


def newtons_global_uneven(XY: Matrix, order: int = None) -> Polynomial:
    coeffs = divdiff_coeffs(XY, order)
    result = Polynomial(coeffs[0])
    coeffs = coeffs[1:]
    for i, k in enumerate(coeffs):
        result += Polynomial.fromroots(XY[: (i + 1), 0]) * k
    return result


def newtons_global_even(
    Y: Vector, start: float, step: float, order: int = None
) -> Polynomial:
    diffs = diff_coeffs(Y, order)
    result = Polynomial(diffs[0])
    for i, d in enumerate(diffs[1:]):
        roots = np.arange(start, start + (i + 1) * step, step)
        coeff = d / factorial(i + 1) / np.power(step, i + 1)
        result += coeff * Polynomial.fromroots(roots)
    return result


def pretty_poly(poly: Polynomial) -> str:
    return str(poly).replace("**", "^").replace(" ", "")


if __name__ == "__main__":

    order = 3
    even_XY = np.array([[0, 1], [1, 2], [2, 4], [3, 3], [4, 1]], dtype=float)
    uneven_XY = np.array(
        [[0.4, -1], [1.3, 1], [1.5, -1], [2, 1], [2.4, 4]], dtype=float
    )
    rule, ran = lambda x: np.sqrt(np.abs(x + np.cos(x) / 2)), np.arange(0, 5, 1)
    
    
    lb, rb = -1, 5
    x_lin = np.linspace(lb, rb, 100)

    poly_even = newtons_global_even(even_XY[:, 1], start=0, step=1, order=order)
    assert poly_even == newtons_global_uneven(even_XY, order)
    print("[even as uneven]", pretty_poly(poly_even))
    plt.scatter(even_XY[:, 0], even_XY[:, 1], label="Even Dots", color="blue")
    ex, ey = poly_even.linspace(n=100, domain=[lb, rb])
    plt.plot(ex, ey, label=f"Even N_{order}", color="blue")

    poly_uneven = newtons_global_uneven(uneven_XY, order)
    print("[uneven]", pretty_poly(poly_uneven))
    plt.scatter(uneven_XY[:, 0], uneven_XY[:, 1], label="Uneven Dots", color="red")
    ux, uy = poly_uneven.linspace(n=100, domain=[lb, rb])
    plt.plot(ux, uy, label=f"Uneven N_{order}", color="red")

    rule_XY = np.array(list(zip(rule(ran), list(ran))), dtype=float)
    poly_rule = newtons_global_uneven(rule_XY, order)
    print("[rule (uneven)]", pretty_poly(poly_rule))
    plt.scatter(rule_XY[:, 0], rule_XY[:, 1], label="Rule Dots", color="green")
    rx, ry = poly_rule.linspace(n=100, domain=[lb, rb])
    plt.plot(rx, ry, label=f"Rule N_{order}", color="green")

    plt.ylim((-2, 7))
    plt.grid()
    plt.legend()
    plt.show()
