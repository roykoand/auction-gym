import numpy as np
from numba import jit


@jit(nopython=True)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def normal_pdf(x: float, mu: float, sigma: float) -> float:
    return np.exp(-(((x - mu) / sigma) ** 2) / 2) / (sigma * np.sqrt(2 * np.pi))
