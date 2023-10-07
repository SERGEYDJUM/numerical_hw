from typing import Callable, Iterable, Self
import numpy as np
from timeit import timeit


class DetSLE:
    def __init__(
        self, coeff_matrix: Iterable[Iterable[float]], free_vec: Iterable[float]
    ) -> Self:
        self.A = np.array(coeff_matrix, dtype=np.float32)
        self.Y = np.fromiter(free_vec, dtype=np.float32)

        # Square check
        assert self.A.shape[0] == self.A.shape[1] == self.Y.shape[0]

        # Gauss method prerequisite
        for i in range(self.A.shape[0]):
            assert self.A[i][i] != 0

        # Fully defined check
        for row in self.A:
            assert not (len(set(row)) == 1 and (row[0] == 0))

        # Jordan-Gauss prerequisite
        for col in self.A.transpose():
            assert not (len(set(col)) == 1 and (col[0] == 0))
            
        # Jacobi prerequisite
        for i, row in enumerate(self.A):
            assert np.sum(np.abs(row)) < 2*abs(row[i])


    def __repr__(self) -> str:
        return f"{self.A} * X = {self.Y}"

    def gauss(self) -> list[float] | None:
        AY = np.column_stack((self.A, self.Y))
        n, _ = AY.shape
        # Forward
        det_A = 1.0
        for i in range(n):
            det_A *= AY[i][i]
            AY[i] /= AY[i][i]
            for j in range(i + 1, n):
                AY[j] -= AY[i] * AY[j][i]

        # Determinant check
        if det_A == 0:
            return None

        # Reverse
        X = np.zeros(n, dtype=np.float32)
        X[-1] = AY[-1][-1]
        for i in range(0, n - 1)[::-1]:
            X[i] = AY[i][-1]
            for j in range(i + 1, n):
                X[i] -= AY[i][j] * X[j]

        return X

    def jordan_gauss(self) -> list[float]:
        AY = np.column_stack((self.A, self.Y))
        n, _ = AY.shape

        # Forward
        for i in range(n):
            if AY[i][i] == 0:
                for j in range(i, n):
                    if AY[j][i] != 0:
                        AY[i], AY[j] = AY[j], AY[i]
                        break

            AY[i] /= AY[i][i]

            for j in range(i + 1, n):
                AY[j] -= AY[i] * AY[j][i]

        # Reverse
        for i in range(n - 1)[::-1]:
            for j in range(i + 1, n):
                # Optimized (changing only Y)
                AY[i][-1] -= AY[j][-1] * AY[i][j]

        return AY[:, -1]

    def jacobi(self, eps=0.01) -> list[float] | None:
        C = self.A.copy()
        D = self.Y.copy()
        n, _ = C.shape
        
        for i in range(n):
            D[i] /= C[i][i]
            C[i] /= C[i][i]
            C[i][i] = 0
        
        X_old, X_new = D + 10*eps, D
        while np.linalg.norm(X_new - X_old) > eps:
            X_new, X_old = D - (C @ X_new), X_new
        
        return X_new
        

def bench_us(f: Callable) -> float:
    return timeit(f, number=1000) * 1000


if __name__ == "__main__":
    sle = DetSLE([[5, 0, 1], [2, 6, -2], [-3, 2, 10]], [11, 8, 6])
    # sle = DetSLE([[9, 3, 1], [4, 2, 1], [1, 1, 1]], [3, 1, 0])
    print(f"Gauss: {sle.gauss()} in {bench_us(sle.gauss):.2f}μs")
    print(f"Jordan-Gauss: {sle.jordan_gauss()} in {bench_us(sle.jordan_gauss):.2f}μs")
    print(f"Simple Iter: {sle.jacobi()} in {bench_us(sle.jacobi):.2f}μs")
