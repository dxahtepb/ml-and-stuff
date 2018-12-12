from abc import abstractmethod
import numpy as np


class Kernel:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _function(self, x, y):
        pass

    def __call__(self, x, y):
        return self._function(x, y)


class Polynomial(Kernel):
    def _function(self, x, y):
        gamma = self.gamma if self.gamma is not None else 1/(2 * len(x))
        return (gamma * np.inner(x, y) + self.coef0)**self.degree

    def __init__(self, degree, coef0=0.0, gamma=None):
        super(Polynomial, self).__init__()
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma


class Linear(Kernel):
    def __init__(self):
        super(Linear, self).__init__()

    def _function(self, x, y):
        return np.dot(x, y)


class RadialBasis(Kernel):
    def __init__(self, sigma):
        super(RadialBasis, self).__init__()
        self.gamma = 1 / (2 * sigma**2)

    def _function(self, x, y):
        return np.exp(
            -self.gamma * np.linalg.norm(np.array(x) - np.array(y))**2
        )


class Tangent(Kernel):
    def _function(self, x, y):
        return np.tanh(np.dot(self.k * np.array(x), y) + self.c)

    def __init__(self, k, c):
        super(Tangent, self).__init__()
        assert k > 0 > c
        self.k = k
        self.c = c
