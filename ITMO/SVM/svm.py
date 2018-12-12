from functools import partial

import numpy as np
import cvxopt

from ITMO.SVM.kernels import Polynomial, Linear
from ITMO.util import silence_stdout


class SVMClassifier:
    def __init__(self, C, kernel, verbose=False):
        self.C = C
        self.kernel = kernel
        self.verbose = verbose
        self.weights = []
        self.supp_classes = []
        self.supp_vectors = []
        self.b = 0

    def _reset(self):
        self.supp_vectors = []
        self.supp_classes = []
        self.weights = []
        self.b = None

    def fit(self, xs, ys):
        """
        using cvxopt quadratic programming solver

        minimize (1/2) * l.T * Q * l + -1 * e.T * x
        subject to A.T * l = b
                   -l <= 0
                   l <= h

        where Q = kernels matrix ((y[i]*y[j]*k(x[i],x[j]), ...), ...)
              A = ys
              l = lambdas (l[i], ...)
              b = 0
              e = vec of ones (1, 1, ...)
              h = C * e
        """
        self._reset()

        xs = np.array(xs)
        ys = np.array(ys)

        q_matrix = []
        for i in range(len(xs)):
            q_matrix.append([])
            for j in range(len(xs)):
                q_matrix[i].append(ys[i] * ys[j] * self.kernel(xs[i], xs[j]))
        q_matrix = np.array(q_matrix)

        e = -1 * np.ones(len(xs))

        a_matrix = np.atleast_2d(ys)
        b = 0.0

        g_matrix = np.diag(-1 * np.ones(len(xs)))
        h = np.array(np.zeros(len(xs)))

        g_matrix = np.vstack((g_matrix, np.diag(np.ones(len(xs)))))
        h = np.append(h, np.ones(len(xs)) * self.C)

        (P, q, G, h, A, b) = list(map(
                partial(cvxopt.matrix, tc='d'),
                (q_matrix, e, g_matrix, h, a_matrix, b)
            ))

        if not self.verbose:
            with silence_stdout():
                solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        else:
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        for idx, l in enumerate(np.ravel(solution['x'])):
            if l < 1e-5:
                continue

            self.supp_vectors.append(xs[idx])
            self.supp_classes.append(ys[idx])
            self.weights.append(l)

        self.supp_vectors, self.supp_classes, self.weights = \
            map(np.array, (self.supp_vectors, self.supp_classes, self.weights))

        if isinstance(self.kernel, (Polynomial, Linear)):
            idx = 0
            while self.b is None:
                if self.weights[idx] + 1e-6 < self.C:
                    self.b = 0
                    for l_i, x_i, y_i in zip(self.weights, self.supp_vectors, self.supp_classes):
                        self.b += l_i * y_i * self.kernel(x_i, self.supp_vectors[idx])
                    self.b -= self.supp_classes[idx]
                idx += 1
        else:
            self.b = 0

    def predict(self, xs):
        ys = list()

        for x in xs:
            res = 0
            for l_i, x_i, y_i in zip(self.weights, self.supp_vectors, self.supp_classes):
                res += l_i * y_i * self.kernel(x_i, x)
            ys.append(res - self.b)

        return list(map(int, np.sign(ys)))
