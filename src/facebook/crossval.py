# Built-in.
from typing import (
    Iterable, Optional, Tuple,
)

# External.
import numpy as np


class CrossValidation:
    def __init__(
            self,
            folds: int = 5,
            shuffle: bool = False,
    ):
        """
        K-Folds cross-validator.

        :param folds: Number of folds.
        :param shuffle: Whether to shuffle the data before splitting into batches.
        """
        self.folds = folds
        self.shuffle = shuffle

    def split(
            self,
            X: np.ndarray,  # noqa
            y: Optional[np.ndarray] = None
    ) -> Iterable[Tuple[int, int]]:
        """
        Generate indices to split data into training and test set.
        """
        samples, _ = X.shape
        indices = np.arange(samples)
        if self.shuffle:
            np.random.shuffle(indices)
        folds = self.folds
        fold_sizes = np.full(folds, samples // folds, dtype=int)
        fold_sizes[:samples % folds] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_index = indices[start:stop]
            test_mask = np.zeros(samples, dtype=bool)
            test_mask[test_index] = True
            train_index = indices[np.logical_not(test_mask)]
            yield train_index, test_index
            current = stop
