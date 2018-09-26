from abc import abstractmethod
from typing import List
from collections import namedtuple

from .dataset import Point
from ITMO.util import argmax


Distance = namedtuple('Distance', 'dist label')


class ClassifierModel:
    @abstractmethod
    def fit(self, points, n_classes):
        pass

    @abstractmethod
    def test_dataset(self, dataset):
        pass

    @abstractmethod
    def test_one(self, point):
        pass


class WeightedKNNClassifier(ClassifierModel):
    def __init__(self, n_neighbors, distance_metric, kernel):
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.kernel = kernel
        self.n_classes = 0
        self.state = None

    def fit(self, points, n_classes):
        self.state = points
        self.n_classes = n_classes

    def test_dataset(self, dataset: List[Point]):
        return [
            Point(self.test_one(point), point.coords)
            for point in dataset
        ]

    def test_one(self, point: Point):
        distances = list()
        for known_point in self.state:
            distances.append(Distance(
                self.distance_metric(point, known_point), known_point.label))
        return self._apply_weights(sorted(distances)[:self.n_neighbors+1])

    def _apply_weights(self, distances):
        possible_labels = [0] * self.n_classes
        max_dist = distances[-1].dist
        for dist in distances:
            possible_labels[dist.label] += self.kernel(dist.dist / max_dist)
        return argmax(possible_labels)
