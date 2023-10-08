from typing import Callable, Self, Tuple
import numpy as np
from timeit import timeit

Vector = np.ndarray[Tuple[int], float]
Matrix = np.ndarray[Tuple[int, int], float]


class DetSLE:
    def __init__(self, coeff_matrix: Matrix, free_vec: Vector) -> Self:
        self.A = np.array(coeff_matrix, dtype=np.float_)
        self.Y = np.fromiter(free_vec, dtype=np.float_)
        self.n = self.A.shape[0]

        # Square check
        assert self.A.shape[0] == self.A.shape[1] == self.Y.shape[0]

        # Fully defined check
        for row in self.A:
            assert not (len(set(row)) == 1 and (row[0] == 0))

        # Gauss-Jordan prerequisite
        for col in self.A.transpose():
            assert not (len(set(col)) == 1 and (col[0] == 0))

        # Try to make diagonally dominant matrix
        P = self._pivot_matrix()
        self.A, self.Y = P @ self.A, P @ self.Y

        # Jacobi prerequisite
        for i, row in enumerate(self.A):
            if np.sum(np.abs(row)) > 2 * abs(row[i]):
                print("Jacobi probably won't converge")
                break

    def __repr__(self) -> str:
        return f"{self.A} * X = {self.Y}"

    def _pivot_matrix(self) -> Matrix:
        A, n = self.A, self.n
        P = np.identity(n, dtype=np.float_)
        for i in range(n):
            max_idx = i + np.argmax(np.abs(A[i:n, i]))
            if max_idx != i:
                P[[i, max_idx]] = P[[max_idx, i]]
        return P

    def _express_unknown(self) -> Tuple[Matrix, Vector]:
        C, D = self.A.copy(), self.Y.copy()
        for i in range(C.shape[0]):
            D[i] /= C[i, i]
            C[i] /= -C[i, i]
            C[i, i] = 0
        return C, D

    # Task 1.1 with 1.3 applied (метод Гаусса)
    def gauss(self) -> Vector | None:
        AY, n = np.column_stack((self.A, self.Y)), self.n
        # Forward
        det_A = 1.0
        for i in range(n):
            # Swap rows to ensure biggest lead
            max_lead = i + np.argmax(np.abs(AY[i:, i]))
            if max_lead != i:
                AY[[max_lead, i]] = AY[[i, max_lead]]

            det_A *= AY[i, i]
            AY[i] /= AY[i, i]
            for j in range(i + 1, n):
                AY[j] -= AY[i] * AY[j, i]

        # Determinant check
        if det_A == 0:
            return None

        # Reverse
        X = np.zeros(n, dtype=np.float_)
        X[-1] = AY[-1, -1]
        for i in range(0, n - 1)[::-1]:
            X[i] = AY[i, -1]
            for j in range(i + 1, n):
                X[i] -= AY[i, j] * X[j]

        return X

    # Task 1.2 with 1.3 applied (метод исключения)
    def gauss_jordan(self) -> Vector:
        AY, n = np.column_stack((self.A, self.Y)), self.n
        # Forward
        for i in range(n):
            # Swap rows to ensure biggest lead
            max_lead = i + np.argmax(np.abs(AY[i:, i]))
            if max_lead != i:
                AY[[max_lead, i]] = AY[[i, max_lead]]

            AY[i] /= AY[i, i]

            for j in range(i + 1, n):
                AY[j] -= AY[i] * AY[j, i]

        # Reverse (affects only answer column)
        for i in range(n - 1)[::-1]:
            for j in range(i + 1, n):
                AY[i, -1] -= AY[j, -1] * AY[i, j]

        return AY[:, -1]

    def _lu_decomposition(self) -> Matrix:
        A, n = self.A, self.n
        LU = np.identity(n, dtype=np.float_)
        for i in range(n):
            for j in range(i):
                LU[j, i] = A[j, i] - LU[:j, i] @ LU[j, :j]
            for j in range(i, n):
                LU[j, i] = (A[j, i] - LU[:i, i] @ LU[j, :i]) / LU[i, i]
        return LU

    # Task 1.5 (метод с LU-декомпозицией)
    def lu_method(self) -> Vector:
        LU = self._lu_decomposition()
        Y, n = self.Y, self.n

        y = np.zeros_like(Y)
        for i in range(n):
            y[i] = Y[i] - LU[i, :i] @ y[:i]

        X = np.zeros_like(Y)
        for i in range(n)[::-1]:
            X[i] = (y[i] - (LU[i, i + 1 : n] @ X[i + 1 : n])) / LU[i, i]

        return X

    # Task 2.1 (метод простых итераций)
    def jacobi(self, eps=0.01) -> Vector:
        C, D = self._express_unknown()
        X_old, X_new = D + 10 * eps, D
        while np.linalg.norm(X_new - X_old) > eps:
            X_new, X_old = D + (C @ X_new), X_new
        return X_new

    # Task 2.2 (метод Зейделя)
    def seidel(self, eps=0.01) -> Vector:
        C, D = self._express_unknown()
        n = self.n

        X_old, X = D + 10 * eps, D
        while np.linalg.norm(X - X_old) > eps:
            X_next = np.zeros(n, dtype=np.float_)
            for i in range(n):
                X_next[i] = D[i] + (C[i, i:] @ X[i:] + C[i, :i] @ X_next[:i])
            X_old, X = X, X_next
        return X


def bench_us(f: Callable) -> float:
    return timeit(f, number=1000) * 1000


if __name__ == "__main__":
    # sle = DetSLE([[5, 0, 1], [2, 6, -2], [-3, 2, 10]], [11, 8, 6])
    sle = DetSLE([[9, 3, 1], [2, 5, 1], [1, 1, 6]], [3, 2, 4])
    # sle = DetSLE([[2, 1, 4], [3, 2, 1], [1, 3, 3]], [16, 10, 16])
    print(
        "System will be automatically transformed to be as diagonally dominant as possible"
    )
    print()
    print(sle)
    print()
    print("Direct methods:")
    print(f"\tGauss Method : {sle.gauss()} in {bench_us(sle.gauss):.2f}μs")
    print(
        f"\tGauss-Jordan : {sle.gauss_jordan()} in {bench_us(sle.gauss_jordan):.2f}μs"
    )
    print(f"\tLU Method    : {sle.lu_method()} in {bench_us(sle.lu_method):.2f}μs")
    print("Iterative methods:")
    print(f"\tSimple Iter  : {sle.jacobi()} in {bench_us(sle.jacobi):.2f}μs")
    print(f"\tSeidel Method: {sle.seidel()} in {bench_us(sle.seidel):.2f}μs")
