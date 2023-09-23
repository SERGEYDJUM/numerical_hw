import numpy as np
from numpy.polynomial.polynomial import Polynomial
from random import uniform, choice

__OG_ITERS = 0
__SI_ITERS = 0
__BR_ITERS = 0
__SE_ITERS = 0


def roots_ring(f: Polynomial) -> (float, float):
    abs_coeffs = np.abs(f.coef)[::-1]
    A = np.max(abs_coeffs[1:])
    B = np.max(abs_coeffs[:-1])
    r = 1 / (1 + B / abs_coeffs[-1])
    R = 1 + A / abs_coeffs[0]
    return (r, R)


def x_init(f: Polynomial, r: float, R: float, max_iter: int = 1024) -> float | None:
    D2f = f.deriv(2)
    for _ in range(max_iter):
        x = choice((uniform(-R, -r), uniform(r, R)))
        if f(x) * D2f(x) > 0:
            return x


def newtons(f: Polynomial, x: float, eps: float = 10**-5) -> float:
    global __OG_ITERS

    Df = f.deriv(1)
    x_prev = x + 3 * eps
    while abs(x - x_prev) > eps:
        x_prev, x = x, x - f(x) / Df(x)
        __OG_ITERS += 1
    return x


def simplified_newtons(f: Polynomial, x: float, eps: float = 10**-5) -> float | None:
    global __SI_ITERS

    Dfx0 = f.deriv(1)(x)
    x_prev = x + 3 * eps
    while abs(x - x_prev) > eps:
        # Check if it diverges
        if abs(f(x) / Dfx0) > 1 / eps:
            return None

        x_prev, x = x, x - f(x) / Dfx0
        __SI_ITERS += 1
    return x


def broyden_newtons(f: Polynomial, x: float, c: float, eps: float = 10**-5) -> float:
    global __BR_ITERS

    Df = f.deriv(1)
    x_prev = x + 3 * eps
    while abs(x - x_prev) > eps:
        x_prev, x = x, x - c * f(x) / Df(x)
        __BR_ITERS += 1
    return x


def secant(f: Polynomial, x: float, dx: float, eps: float = 10**-5) -> float:
    global __SE_ITERS

    x_prev = x + 3 * eps
    deriv = (f(x) - f(x - dx)) / dx
    while abs(x - x_prev) > eps:
        x_prev, x = x, x - f(x) / deriv
        deriv = (f(x) - f(x_prev)) / (x - x_prev)
        __SE_ITERS += 1
    return x


if __name__ == "__main__":
    eps = 0.001
    c = 1.15
    dx = 0.1
    poly = Polynomial([-7, 10, -8, -4, 3])  # -7 + 10x - 8x^2 - 4x^3 + 3x^4
    print(f"Input polynomial: {poly}")

    r, R = roots_ring(poly)
    print(f"Roots are in ({-R:.2f}; {-r:.2f}) and ({r:.2f}; {R:.2f})")

    x0 = x_init(poly, r, R)
    print(f"Initial guess: {x0:.3f} \n")

    nr = newtons(poly, x0, eps)
    print(f"Newton's method in {__OG_ITERS} iters: {nr:.5f}")

    snr = simplified_newtons(poly, x0, eps)
    if snr is not None:
        print(f"Simplified Newton's method in {__SI_ITERS} iters: {snr:.5f}")
    else:
        print(f"Simplified Newton's method failed to converge")

    nbr = broyden_newtons(poly, x0, c, eps)
    print(f"Newton-Broyden's method in {__BR_ITERS} iters: {nbr:.5f}")

    ser = secant(poly, x0, dx, eps)
    print(f"Secants method in {__SE_ITERS} iters: {ser:.5f} \n")
