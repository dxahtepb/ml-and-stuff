import random
from abc import abstractmethod

import numpy as np


class OptimizationMethod:
    def __init__(self, cost_function, max_iterations, convergence, verbose):
        self._cost_function = cost_function
        self._max_iterations = max_iterations
        self._convergence = convergence
        self._verbose = verbose

    @abstractmethod
    def optimize(self, xs, ys, initial_weights):
        pass

    @staticmethod
    @abstractmethod
    def init_weights(points):
        pass


class GradientDescend(OptimizationMethod):
    def __init__(self, cost_function, learning_rate, max_iterations=1000,
                 convergence=1e-8, verbose=False):
        super(GradientDescend, self).__init__(cost_function, max_iterations,
                                              convergence, verbose)
        self._learning_rate = learning_rate

    @staticmethod
    def init_weights(points):
        return np.zeros(points[0].coords.size + 1, np.double)

    def optimize(self, xs, ys, initial_weights):
        xs = np.column_stack((np.ones(xs.shape[0]).transpose(), xs))
        weights = initial_weights

        for iteration in range(self._max_iterations):
            ys_predicted = np.array([np.dot(weights, point) for point in xs])
            gradient = np.array([
                np.dot(ys_predicted - ys, xs[:, i]) / xs.size
                for i in range(xs.shape[1])
            ])

            if self._verbose and iteration % 50 == 0:
                print(f'{iteration}: {weights}')

            prev_weights = weights
            prev_cost = self._cost_function(ys, [np.dot(prev_weights, x) for x in xs])

            weights = weights - self._learning_rate * gradient
            new_cost = self._cost_function(ys, [np.dot(weights, x) for x in xs])

            if abs(new_cost - prev_cost) < self._convergence:
                if self._verbose:
                    print(f'Stopped at iteration {iteration}: {weights}')
                break

        return weights


class DifferentialEvolution(OptimizationMethod):
    def __init__(self, cost_function, population_size,
                 differential_weight=0.8, crossover_probability=0.9,
                 max_iterations=1000, convergence=1e-8, verbose=False):
        super(DifferentialEvolution, self).__init__(cost_function, max_iterations,
                                                    convergence, verbose)
        self._p_size = population_size
        self._cp = crossover_probability
        self._f = differential_weight

    @staticmethod
    def init_weights(points):
        return [0] * (len(points[0].coords) + 1)

    def optimize(self, xs, ys, initial_weights):
        xs = np.column_stack((np.ones(xs.shape[0]).transpose(), xs))

        population = np.array([
            [random.random() for _ in range(len(initial_weights))]
            for _ in range(self._p_size)
        ])

        costs = [self._cost_function(ys, [np.dot(w, x) for x in xs]) for w in population]

        solution_candidate = population[np.argmin(costs)]

        for iteration in range(self._max_iterations):
            if self._verbose and iteration % 100 == 0:
                print(f'{iteration}: {solution_candidate}')

            for idx, w in enumerate(population):
                w_new = np.copy(w)

                population_wo_w = [vec for j, vec in enumerate(population) if j != idx]
                a, b, c = random.sample(population_wo_w, 3)

                cut_point = random.randint(0, w.size-1)
                for j, (ai, bi, ci) in enumerate(zip(a, b, c)):
                    if random.random() < self._cp or j == cut_point:
                        w_new[j] = ai + self._f*(bi - ci)

                cost = self._cost_function(ys, [np.dot(w_new, x) for x in xs])

                if cost < costs[idx]:
                    population[idx] = w_new
                    costs[idx] = cost

            solution_candidate = population[np.argmin(costs)]

        return solution_candidate
