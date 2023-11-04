import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

Matrix = np.ndarray[(int, int), float]
Vector = np.ndarray[(int), float]


def global_lagrange(XY: Matrix) -> Polynomial:
    n = len(XY) - 1
    poly = Polynomial((0))
    for i in range(len(XY)):
        roots = np.delete(XY[:, 0], i)
        denom = np.prod(np.full(n, XY[i, 0]) - roots)
        poly += Polynomial.fromroots(roots) * XY[i, 1] / denom
    return poly


def linear_lagrange(XY: Matrix, arr: Vector) -> Vector:
    XY = XY[XY[:, 0].argsort()]
    binned = np.digitize(arr, XY[:, 0], True)
    inferred = []
    for i in range(len(arr)):
        x_l, y_l = XY[binned[i] - 1]
        x_r, y_r = XY[binned[i]]
        p_l = (arr[i] - x_r) / (x_l - x_r)
        p_r = (arr[i] - x_l) / (x_r - x_l)
        inferred.append(p_l * y_l + p_r * y_r)

        assert abs(p_l + p_r - 1) < 1e-9

    return np.array(inferred, dtype=np.float_)


def lagrange_cubic_coeff(x, l, m, r) -> (float, float, float):
    p_1 = (x - m) * (x - r) / ((l - m) * (l - r))
    p_2 = (x - l) * (x - r) / ((m - l) * (m - r))
    p_3 = (x - l) * (x - m) / ((r - l) * (r - m))
    assert abs(p_1 + p_2 + p_3 - 1) < 1e-9
    return (p_1, p_2, p_3)


def cubic_lagrange(XY: Matrix, arr: Vector) -> Vector:
    XY = XY[XY[:, 0].argsort()]
    binned = np.digitize(arr, XY[:, 0], True)
    inferred = []
    for i in range(len(arr)):
        x_2, y_2 = XY[binned[i] - 1]
        x_3, y_3 = XY[binned[i]]
        l1, l2 = None, None

        if binned[i] - 2 >= 0:
            x_1, y_1 = XY[binned[i] - 2]
            p_11, p_12, p_13 = lagrange_cubic_coeff(arr[i], x_1, x_2, x_3)
            l1 = p_11 * y_1 + p_12 * y_2 + p_13 * y_3
            l2 = l1

        if binned[i] + 1 < XY.shape[0]:
            x_4, y_4 = XY[binned[i] + 1]
            p_22, p_23, p_24 = lagrange_cubic_coeff(arr[i], x_2, x_3, x_4)
            l2 = p_22 * y_2 + p_23 * y_3 + p_24 * y_4
            if not l1:
                l1 = l2

        inferred.append((l1 + l2) / 2)

    return np.array(inferred, dtype=np.float_)


if __name__ == "__main__":
    dots = np.array([(1, 4), (1.5, -6), (2.5, 0), (4.5, -3)])
    poly = global_lagrange(dots)
    print(f"[Global] {poly}")

    lb, rb = 1, 4.5

    x_lin = np.linspace(lb, rb, 100)
    poly_y = poly.linspace(n=100, domain=[lb, rb])

    plt.scatter(dots[:, 0], dots[:, 1], label="Data", color="black")
    plt.plot(poly_y[0], poly_y[1], label="Global")
    plt.plot(x_lin, linear_lagrange(dots, x_lin), label="Linear")
    plt.plot(x_lin, cubic_lagrange(dots, x_lin), label="Cubic")
    plt.grid()
    plt.legend()
    plt.show()
