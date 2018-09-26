from typing import List, Any, Sequence


class Point:
    def __init__(self, label: Any, coords: Sequence):
        self.coords = coords
        self.label = label

    def __str__(self):
        return \
            f'(coords: {" ".join(map(str, self.coords))}, label: {self.label})'


class Dataset(List[Point]):
    pass
