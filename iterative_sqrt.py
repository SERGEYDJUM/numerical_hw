from random import uniform

def simple_iteration(
    a: float,
    x: float = uniform(-20.0, 21.0),
    eps: float = 10.0**-5,
    max_iters: int = 128,
) -> float | None:
    for _ in range(max_iters):
        x_next = abs(x + a/x)/2
        if abs(x_next - x) <= eps:
            return x_next
        x = x_next
    return x_next

if __name__ == "__main__":
    ans = simple_iteration(5)
    print(f"Iterative approximation of sqrt(5): {ans}")
    print(f"Error (0.0 if too little): {abs(5**0.5 - ans)}")