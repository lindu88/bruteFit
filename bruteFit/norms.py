from abc import ABC, abstractmethod
import numpy as np

class Normalizer(ABC):
    @abstractmethod
    def norm(self, x):
        pass

    @abstractmethod
    def inv(self, x):
        pass

class MaxNorm(Normalizer):
    def __init__(self, y):
        if y is None:
            self.max_y = 1
        else:
            y = np.asarray(y, dtype=float)
            self.max_y = np.nanmax(np.abs(y)) or 1e-12

    def norm(self, x):
        x = np.asarray(x, dtype=float)
        return x / self.max_y

    def inv(self, x):
        x = np.asarray(x, dtype=float)
        return x * self.max_y

    def set_y(self, y):
        y = np.asarray(y, dtype=float)
        self.max_y = np.nanmax(np.abs(y)) or 1e-12