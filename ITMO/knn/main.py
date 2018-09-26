from ITMO.knn.dataset import Point
from ITMO.util import read_csv


def make_dataset(table):
    samples = list()
    for row in table:
        samples.append(Point(
            int(row[-1]),
            list(map(float, row[:-1])),
        ))
    return samples


def main():
    table = read_csv('dataset.txt', '\t')
    all_samples = make_dataset(table)


if __name__ == '__main__':
    main()
