import os
import sys
from contextlib import contextmanager

import numpy as np


def argmax(sequence):
    max_possible = -1
    max_label = 0
    for idx, x in enumerate(sequence):
        if x > max_possible:
            max_possible = x
            max_label = idx
    return max_label


def read_csv(file_name, delim, skip_header=False):
    lines = list()
    for line in open(file_name, 'r'):
        lines.append(line.rstrip().split(delim))
    return lines[1 if skip_header else 0:]


def normalize(xs: np.ndarray):
    mean = sum(xs) / len(xs)
    deviation = (sum((xs - mean)**2) / len(xs)) ** (1/2)
    return (xs - mean) / deviation, mean, deviation


def denormalize(xs, mean, deviation):
    return xs * deviation + mean


@contextmanager
def silence_stdout():
    new_target = open(os.devnull, 'w')
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield
    finally:
        sys.stdout = old_target
