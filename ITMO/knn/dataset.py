from typing import List, Any, Sequence


class Point:
    def __init__(self, label: Any, coords: Sequence):
        self.coords = coords
        self.label = label

    def __str__(self):
        return \
            '(coords: {}, label: {})'.format(" ".join(map(str, self.coords)), self.label)
