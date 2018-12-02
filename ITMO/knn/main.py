import random
import logging
import copy

from ITMO.knn.statistics import Metrics

import ITMO.knn.functions.kernel as kernels
import ITMO.knn.functions.distance as distances

from ITMO.knn.dataset import Point
from ITMO.knn.knn import WeightedKNNClassifier
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


def leave_one_out_validation(samples, classifier, n_classes):
    misses = 0
    for idx in range(len(samples)):
        one_out_data = copy.copy(samples)
        sample = one_out_data.pop(idx)

        classifier.fit(one_out_data, n_classes)
        predicted_label = classifier.test_one(sample)
        if predicted_label != sample.label:
            misses += 1
    return misses


def evaluate_optimal_k(samples, classifier_class, n_classes, k_limit,
                       **classifier_opts):
    leave_i_out = []

    for k in range(1, k_limit):
        misses = leave_one_out_validation(
            samples,
            classifier_class(**classifier_opts, n_neighbors=k),
            n_classes,
        )

        leave_i_out.append((misses, k))

    logging.debug('%s', ', '.join(map(str, leave_i_out)))

    return min(leave_i_out)[1]


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


def k_fold_cross_validation(samples, classifier, n_classes,
                            accuracy_measure, k_fold=10, n_tries=1):
    accuracy_sum = 0

    for try_num in range(1, n_tries+1):
        random.shuffle(samples)

        for k in range(k_fold):
            train_data, test_data = split_dataset_k_fold(samples, k_fold, k)
            classifier.fit(train_data, n_classes=n_classes)
            predicted_data = classifier.test_dataset(test_data)
            accuracy_sum += accuracy_measure(test_data, predicted_data)

        logging.debug('Try №%d: %f', try_num, accuracy_sum / k_fold / try_num)

    return accuracy_sum / k_fold / n_tries


N_NEIGHBORS = 10
DISTANCE_METRIC = distances.manhattan
KERNEL = kernels.sigmoid


def main():
    table = read_csv('dataset.txt', '\t')
    all_samples = make_dataset(table)
    classifier = WeightedKNNClassifier(N_NEIGHBORS, DISTANCE_METRIC, KERNEL)
    x = k_fold_cross_validation(all_samples, classifier, n_classes=2, k_fold=9,
                                accuracy_measure=Metrics.f_score)
    print(x)


if __name__ == '__main__':
    # random.seed(42)
    logging.basicConfig(level=logging.DEBUG)
    main()
