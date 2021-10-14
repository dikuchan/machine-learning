# Built-in.
from typing import (
    Optional,
)

# External.
import numpy as np


class StandardScaler:
    def __init__(self):
        self._mean: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None

    def fit(
            self,
            X: np.ndarray,  # noqa
            y: Optional[np.ndarray] = None,
    ) -> "StandardScaler":
        """
        Compute the mean and std to be used for later scaling.
        """
        rows, cols = X.shape
        if y is not None:
            yrows, = y.shape
            assert yrows == rows, "no. of rows of `X` and `y` should match"
        self._mean = X.mean(axis=0)
        self._scale = X.std(axis=0)

        return self

    def transform(
            self,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
    ) -> np.ndarray:  # noqa
        """
        Perform standardization by centering and scaling.
        """
        X -= self._mean
        X /= self._scale

        return X
