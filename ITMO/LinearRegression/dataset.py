import numpy as np
from typing import Union, List


class Point:
    def __init__(self, target: Union[int, float], coords: Union[np.ndarray, List]):
        if isinstance(coords, List):
            self.coords = np.array(coords)
        else:
            self.coords = coords
        self.target = target

    def __str__(self):
        return \
            f'<coords: ({", ".join(map(str, self.coords))}), ' + \
            f'target: {self.target}>'

    def __repr__(self):
        return \
            f'<coords: ({", ".join(map(str, self.coords))}), ' + \
            f'target: {self.target}>'
