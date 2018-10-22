import numpy as np

from ITMO.LinearRegression.dataset import Point
from ITMO.LinearRegression.optimization_method import OptimizationMethod
from ITMO.util import normalize, denormalize


class LinearRegression:
    def __init__(self, optimization_method: OptimizationMethod, normalize=False):
        self._optimization_method = optimization_method
        self._normalize = normalize
        self._weights = None
        self._mean = {'x': None, 'y': 0}
        self._deviation = {'x': None, 'y': 1}

    def fit(self, points):
        self._weights = self._optimization_method.init_weights(points)
        self._weights = self._optimization_method.optimize(
            *self._normalize_data(points), self._weights)

    def _normalize_data(self, points):
        xs = np.array([point.coords for point in points])
        ys = np.array([point.target for point in points])

        self._mean['x'] = [0] * (xs.shape[1])
        self._deviation['x'] = [1] * (xs.shape[1])

        if self._normalize:
            for idx in range(xs.shape[1]):
                xs[:, idx], self._mean['x'][idx], self._deviation['x'][idx] = \
                    normalize(xs[:, idx])
            ys, self._mean['y'], self._deviation['y'] = normalize(ys)

        return xs, ys

    def get_weights(self):
        return self._weights

    def predict_coords(self, xs):
        return [self.predict_one(x) for x in xs]

    def predict_dataset(self, points):
        return [Point(self.predict_one(point.coords), point.coords) for point in points]

    def predict_one(self, xs):
        if self._weights is None:
            raise Exception('Fit the model first')

        xs = (xs - self._mean['x']) / self._deviation['x']
        prediction = np.dot(self._weights[1:], xs) + self._weights[0]
        return denormalize(prediction, self._mean['y'], self._deviation['y'])
