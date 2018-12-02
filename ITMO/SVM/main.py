import random

import numpy as np

from ITMO.SVM import kernels
from ITMO.SVM.svm import SVMClassifier
from ITMO.NewShinyNaiveBayes.Metrics import Metrics
from ITMO.util import read_csv, normalize


def make_dataset(table, shuffle=False):
    samples_x = list()
    samples_labels = list()
    if shuffle:
        np.random.shuffle(table)
    for row in table:
        samples_x.append(list(map(float, row[:-1])))
        samples_labels.append(int(row[-1]))
    return samples_x, samples_labels


def split_dataset(samples, ratio):
    random.shuffle(samples)
    train_size = int(len(samples) * ratio)
    return samples[:train_size], samples[train_size:]


def k_fold_cross_validation(samples, classifier,
                            accuracy_measure, k_fold=10,
                            verbose=False):
    def split_dataset_k_fold(xs, ys, k_fold, k):
        train_data_x = []
        train_data_y = []
        test_data_x = []
        test_data_y = []

        from_ = k * len(xs) // k_fold
        to = (k + 1) * len(xs) // k_fold

        for idx, (x, y) in enumerate(zip(xs, ys)):
            if from_ <= idx < to:
                test_data_x.append(x)
                test_data_y.append(y)
            else:
                train_data_x.append(x)
                train_data_y.append(y)

        return (train_data_x, train_data_y), (test_data_x, test_data_y)

    accuracy_sum = 0
    xs, ys = samples

    for k in range(k_fold):
        train_data, test_data = split_dataset_k_fold(xs, ys, k_fold, k)
        classifier.fit(train_data[0], train_data[1])
        predicted_data = classifier.predict(test_data[0])
        if verbose:
            pred = [max(0, y) for y in predicted_data]
            real = [max(0, y) for y in test_data[1]]
            print('k:', k, real, pred, accuracy_measure(test_data[1], predicted_data))
        accuracy_sum += accuracy_measure(test_data[1], predicted_data)

    return accuracy_sum / k_fold


KERNEL = kernels.Linear


def main():
    table = read_csv('../knn/dataset.txt', '\t')

    xs, ys = make_dataset(table, shuffle=True)
    xs, _, _ = list(normalize(np.array(xs)))
    ys = [-1 if y == 0 else 1 for y in ys]

    classifier = SVMClassifier(C=1000, kernel=kernels.Polynomial(2, gamma=1))
    ans = k_fold_cross_validation((xs, ys), classifier, Metrics.f_score2, k_fold=9, verbose=True)
    print(ans)


if __name__ == '__main__':
    # random.seed(42)
    main()
