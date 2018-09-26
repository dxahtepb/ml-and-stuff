def argmax(sequence):
    max_possible = -1
    max_label = 0
    for idx, x in enumerate(sequence):
        if x > max_possible:
            max_possible = x
            max_label = idx
    return max_label


def read_csv(file_name, delim):
    lines = list()
    for line in open(file_name, 'r'):
        lines.append(line.rstrip().split(delim))
    return lines
