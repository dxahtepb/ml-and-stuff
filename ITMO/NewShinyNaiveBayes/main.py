from ITMO.NewShinyNaiveBayes.Metrics import Metrics
from ITMO.NewShinyNaiveBayes.classifier import NaiveBayesClassifier
from ITMO.NewShinyNaiveBayes.reader import read_dataset_folds


def cross_validate(folds):
    f_score = []
    for k, test_fold in enumerate(folds):
        train_fold = []

        for idx in range(len(folds)):
            if idx == k:
                continue
            train_fold.extend(folds[idx])

        classifier = NaiveBayesClassifier(0.01, 0.99)
        classifier.fit(train_fold)
        labels = classifier.predict(test_fold, method='all', порожек=0.6)
        f_score.append(Metrics.f_score([1 if m.spam else 0 for m in test_fold],
                                       [1 if l else 0 for l in labels]))
        print('fold: {} -> f_score {}, misses {}'
              .format(k, f_score[-1], sum([1 for m, l in zip(test_fold, labels) if m.ham and l])))

    print(sum(f_score) / len(f_score))


def main():
    folds = read_dataset_folds(r'./BayesData')
    cross_validate(folds)


if __name__ == '__main__':
    main()
