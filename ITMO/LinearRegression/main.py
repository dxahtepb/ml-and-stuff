import random
import logging
import numpy as np
import copy

from ITMO.LinearRegression.cost_functions import mse, rmse
from ITMO.LinearRegression.dataset import Point
from ITMO.LinearRegression.optimization_method import (GradientDescend,
                                                       DifferentialEvolution)
from ITMO.LinearRegression.regression import LinearRegression
from ITMO.util import read_csv


def make_dataset(table):
    samples = list()
    for row in table:
        samples.append(Point(
            int(row[-1]),
            list(map(float, row[:-1])),
        ))
    return samples


def split_dataset(samples, ratio):
    random.shuffle(samples)
    train_size = int(len(samples) * ratio)
    return samples[:train_size], samples[train_size:]


def leave_one_out_validation(samples, model, accuracy_measure):
    predicted = []
    for idx in range(len(samples)):
        one_out_data = copy.copy(samples)
        sample = one_out_data.pop(idx)
        model.fit(one_out_data)
        predicted_target = model.predict_one(sample.coords)
        predicted.append(Point(predicted_target, sample.coords))
    return accuracy_measure(predicted, samples)


def split_dataset_k_fold(samples, k_fold, k):
    train_data = []
    test_data = []

    from_ = k * len(samples) // k_fold
    to = (k + 1) * len(samples) // k_fold

    for idx, sample in enumerate(samples):
        if from_ <= idx < to:
            test_data.append(sample)
        else:
            train_data.append(sample)

    assert len(test_data) + len(train_data) == len(samples)
    return train_data, test_data


def k_fold_cross_validation(samples, model, accuracy_measure, k_fold=10):
    accuracy_sum = 0

    random.shuffle(samples)

    for k in range(k_fold):
        train_data, test_data = split_dataset_k_fold(samples, k_fold, k)
        model.fit(train_data)
        predicted_data = model.predict_dataset(test_data)
        accuracy_sum += accuracy_measure(test_data, predicted_data)

    return accuracy_sum / k_fold


def main():
    all_samples = make_dataset(read_csv('data.csv', ',', skip_header=True))

    model1 = LinearRegression(GradientDescend(cost_function=mse,
                                              learning_rate=3,
                                              max_iterations=1000,
                                              convergence=1e-8),
                              normalize=True)
    model2 = LinearRegression(DifferentialEvolution(cost_function=mse,
                                                    population_size=8,
                                                    differential_weight=0.8,
                                                    crossover_probability=0.9,
                                                    max_iterations=300,
                                                    convergence=1e-8),
                              normalize=True)

    model1.fit(all_samples)
    self_predict = model1.predict_dataset(all_samples)
    print(np.sqrt(mse(all_samples, self_predict)))

    model2.fit(all_samples)
    self_predict = model2.predict_dataset(all_samples)
    print(rmse(all_samples, self_predict))

    print(k_fold_cross_validation(all_samples, model1, rmse, 5))


if __name__ == '__main__':
    random.seed(420)
    logging.basicConfig(level=logging.DEBUG)
    main()
