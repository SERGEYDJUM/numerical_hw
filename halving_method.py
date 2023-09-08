from typing import Callable, Tuple
from random import uniform


def dih(
    f: Callable[[float], float],
    a: float,
    b: float,
    eps: float = 10**-5,
    max_iters: int = 128,
) -> float | None:
    f_a, f_c, f_b = 0.0, 0.0, 0.0
    for _ in range(max_iters):
        c = (a + b) / 2
        f_a, f_c, f_b = f(a), f(c), f(b)
        if abs(f_a - f_b) <= eps:
            return c
        if f_a * f_c <= 0.0:
            b = c
        else:
            a = c


def get_init_bounds(
    f: Callable[[float], float], a: float = -20, b: float = 21, iters: int = 100
) -> Tuple[float, float] | None:
    for _ in range(iters):
        x_1 = uniform(a, b)
        x_2 = uniform(a, b)
        if f(x_1) * f(x_2) <= 0.0:
            return x_1, x_2


if __name__ == "__main__":
    func = lambda x: (x - 2) * ((x - 3) ** 2) * (x - 7)
    a, b = get_init_bounds(func)
    print(f"Initial range: ({a:.2f}, {b:.2f})")
    print(f"Solution: {dih(func, a, b)}")
