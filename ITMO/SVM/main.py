import random

import scipy

import ITMO.knn.functions.kernel as kernels_knn
import ITMO.knn.functions.distance as distances
import numpy as np

from ITMO.SVM import kernels
from ITMO.SVM.svm import SVMClassifier
from ITMO.NewShinyNaiveBayes.Metrics import Metrics
from ITMO.knn.dataset import Point
from ITMO.knn.knn import WeightedKNNClassifier
from ITMO.util import read_csv, normalize

from sklearn import svm


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


def count_bicentral(samples, bic_a, mean=False):
    bicentral_samples = []
    means = [0 for _ in samples[0]]
    if mean:
        for idx in range(len(means)):
            means[idx] = np.mean([sample[idx] for sample in samples])
            for sample in samples:
                sample.coords[idx] -= means[idx]
    for sample in samples:
        bicentral_samples.append(
            [np.sqrt(sum([pow(coord - bic_a,2) for coord in sample])),
             np.sqrt(sum([pow(coord + bic_a,2) for coord in sample]))])
    return bicentral_samples


def calc_wilcoxon(samples, classifier_svm: SVMClassifier, classifier_knn: WeightedKNNClassifier,
                    accuracy_measure, k_fold=10,
                    verbose=False, alpha=0.05):

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

    def convert2points(data):
        points = []
        for coords, label in zip(data[0], data[1]):
            points.append(Point(0 if label == -1 else 1, coords))
        return points

    will_acc_knn = []
    will_acc_svm = []
    accuracy_sum_knn = 0
    accuracy_sum_svm = 0
    xs, ys = samples

    for k in range(k_fold):
        train_data, test_data = split_dataset_k_fold(xs, ys, k_fold, k)
        train_data_knn, test_data_knn = convert2points(train_data), convert2points(test_data)

        classifier_svm.fit(train_data[0], train_data[1])
        classifier_knn.fit(train_data_knn, 2)

        predicted_data_svm = classifier_svm.predict(test_data[0])
        predicted_data_knn = classifier_knn.test_dataset(test_data_knn)

        predicted_data_knn = [-1 if y.label == 0 else 1 for y in predicted_data_knn]
        test_data_knn = [-1 if y.label == 0 else 1 for y in test_data_knn]

        if verbose:
            pred = [max(0, y) for y in predicted_data_svm]
            real = [max(0, y) for y in test_data[1]]

            print('knn k:', k, accuracy_measure(test_data_knn, predicted_data_knn))
            print('svm k:', k, accuracy_measure(test_data[1], predicted_data_svm))
            print()

        accuracy_sum_knn += accuracy_measure(test_data_knn, predicted_data_knn)
        accuracy_sum_svm += accuracy_measure(test_data[1], predicted_data_svm)

        will_acc_knn.append(accuracy_measure(test_data_knn, predicted_data_knn))
        will_acc_svm.append(accuracy_measure(test_data[1], predicted_data_svm))

    values = []
    abs_values = []

    for knn, svm_meas in zip(will_acc_knn, will_acc_svm):
        values.append(knn-svm_meas)
        abs_values.append(np.abs(knn-svm_meas))

    all_values = sorted(zip(abs_values, values), key=lambda _val: _val[0])
    cut_values = []
    cnt = 0
    cnt_pos = 0
    cnt_neg = 0

    mean_svm, var_svm = np.mean(will_acc_svm), np.var(will_acc_svm)
    mean_knn, var_knn = np.mean(will_acc_knn), np.var(will_acc_knn)

    print(mean_svm, var_svm)

    mean_svm, var_svm, mean_knn, var_svm = 0, 1, 0, 1

    p = 1 - alpha

    crit_value_svm = mean_svm+np.sqrt(2*var_svm)*scipy.special.erfinv(2*p-1)
    crit_value_knn = mean_knn+np.sqrt(2*var_knn)*scipy.special.erfinv(2*p-1)
    print((crit_value_knn+crit_value_svm)/2)

    for val in all_values:
        if val[0] != 0:
            if not cut_values or val[0] != cut_values[-1][0]:
                cnt += 1
            cut_values.append([val[0], val[1], cnt])
            if val[1] < 0:
                cnt_neg += 1
            else:
                cnt_pos += 1
    ch_sum = 0
    for val in cut_values:
        if cnt_pos > cnt_neg:
            if val[1] < 0:
                ch_sum += val[2]
        else:
            if val[1] > 0:
                ch_sum += val[2]
    print("Willcoxon: ", ch_sum)
    print('knn: {}\nsvm: {}'.format(accuracy_sum_knn/k_fold, accuracy_sum_svm/k_fold))


N_NEIGHBORS = 10
DISTANCE_METRIC = distances.euclidean
KERNEL = kernels_knn.epanechnikov


def main():
    table = read_csv('../knn/dataset.txt', '\t')

    xs, ys = make_dataset(table, shuffle=True)
    # xs, _, _ = list(normalize(np.array(xs)))
    xs = count_bicentral(xs, 0.25)
    ys = [-1 if y == 0 else 1 for y in ys]

    # classifier = svm.SVC(C=50, gamma='scale', )
    # classifier.fit(xs, ys)

    classifier = WeightedKNNClassifier(N_NEIGHBORS, DISTANCE_METRIC, KERNEL)
    # classifier2 = SVMClassifier(C=5, kernel=kernels.Polynomial(2, gamma=1, coef0=1))
    classifier2 = SVMClassifier(C=5, kernel=kernels.RadialBasis(sigma=0.3))

    # classifier = SVMClassifier(C=2, kernel=kernels.Polynomial(2, gamma=(1/len(xs))))
    # ans = k_fold_cross_validation((xs, ys), classifier2, Metrics.f_score2, k_fold=9, verbose=False)
    # print(ans)
    calc_wilcoxon((xs, ys), classifier_knn=classifier, classifier_svm=classifier2
                  ,accuracy_measure=Metrics.f_score2, k_fold=9, verbose=True)


if __name__ == '__main__':
    np.random.seed(42)
    main()
