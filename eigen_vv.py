import numpy as np
from numpy.linalg import norm

Vector = np.ndarray[(int), float]
Matrix = np.ndarray[(int, int), float]


def iterative(
    A: Matrix, X: Vector, eps: float = 1e-9, max_iter: int = 100
) -> (float, Vector):
    X_old, X = X / norm(X), X
    lam_old, lam = 0.0, 0.0
    for i in range(max_iter):
        X_old, X = X, A @ X

        max_idx = np.argmax(np.abs(X_old))
        lam_old, lam = lam, X[max_idx] / X_old[max_idx]

        X /= norm(X)
        if abs(lam_old - lam) < eps:
            break
    return (lam, X)


def rotation(
    A: Matrix, eps: float = 1e-9, max_iter: int = 100
) -> list[(float, Vector)]:
    n = A.shape[0]
    H_res = np.identity(n)
    for itr in range(max_iter):
        small = True
        aa = abs(A[0, -1])
        i_a, j_a = 0, -1

        for i in range(n):
            for j in range(n):
                if i != j:
                    el_abs = abs(A[i, j])

                    if el_abs > eps:
                        small = False

                    if i < j and el_abs > aa:
                        aa = el_abs
                        i_a, j_a = i, j

        if small:
            break

        phi = np.arctan(2 * A[i_a, j_a] / (A[i_a, i_a] - A[j_a, j_a])) / 2

        H = np.identity(n)
        H[i_a, i_a] = np.cos(phi)
        H[j_a, j_a] = H[i_a, i_a]
        H[i_a, j_a] = -np.sin(phi)
        H[j_a, i_a] = -H[i_a, j_a]
        H_res = H_res @ H

        A = np.transpose(H) @ (A @ H)

    return list(zip(np.diag(A), np.transpose(H_res)))


if __name__ == "__main__":
    A = np.array([[5, 1, 2], [1, 4, 1], [2, 1, 3]], dtype=np.float_)
    X = np.array([1, 1, 1], dtype=np.float_)

    it_res = iterative(A, X)
    print("Iterative:")
    print(f"\tλ = {it_res[0]}, X_λ = {list(it_res[1])}")

    print("Rotation:")
    for i, (lam, v) in enumerate(rotation(A)):
        print(f"\tλ_{i+1} = {lam}, X_λ_{i+1} = {list(v)}")
