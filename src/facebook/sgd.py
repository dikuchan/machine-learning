# Internal.
import logging
from enum import auto, Enum
from typing import (
    Any, List, Optional,
)

# External.
import numpy as np  # noqa: F401

__all__ = [
    "Penalty",
    "SGDRegressor",
]

logger = logging.getLogger("machine_learning.facebook.sgd")


class Penalty(Enum):
    """
    L1- and L2-regularization.
    """
    L1 = auto()
    L2 = auto()

    def compute(self, alpha: float, weight: float) -> float:
        penalty = {
            Penalty.L1: alpha if weight > 0. else -alpha,
            Penalty.L2: 2. * alpha * weight,
        }[self]

        return penalty


class SGDRegressor:
    def __init__(
            self,
            iterations: int = 1000,
            alpha: float = 1e-4,
            tolerance: float = 1e-3,
            stumble: int = 6,
            penalty: Optional[Penalty] = Penalty.L2,
            shuffle: bool = True,
            verbose: bool = False,
            eta: float = 1e-2,
    ):
        """
        Stochastic gradient descent.

        :param iterations: Maximum number of passes over the training data.
        :param alpha: Multiplication constant of a regularization term.
        :param tolerance: Stop criterion.
                          Training will finish when ``error > best_error - tolerance``.
        :param stumble: No. of iterations with no improvement to wait before early stop.
        :param penalty: Penalty (aka regularization term).
        :param shuffle: Whether training data should be shuffled on each epoch.
        :param verbose: Logging level.
        :param eta: Initial learning rate.
        """
        self.iterations = iterations
        self.alpha = alpha
        self.tolerance = tolerance
        self.stumble = stumble
        self.penalty = penalty
        self.shuffle = shuffle
        self.verbose = verbose
        self.eta = eta
        # Internal.
        self._bias: Optional[float] = None
        self._weights: Optional[np.ndarray] = None
        self._best_loss = float("inf")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGDRegressor":  # noqa
        """
        Fit a linear model with stochastic gradient descent.

        :param X: Train matrix filled with `n` observations and `p` features (n x p).
        :param y: Target vector of the matrix `X` (n x 1).
        """
        rows, cols = X.shape
        yrows, = y.shape
        assert yrows == rows, "no. of rows of `X` and `y` should match"
        self._bias = 0.
        self._weights = np.zeros(cols, dtype=float)

        # Scaling factor for `eta`.
        t: int = 0
        # Number of times loss doesn't consequentially improve.
        stumble: int = 0

        for e in range(self.iterations):
            if self.shuffle:
                np.random.shuffle(X)
            loss: float = 0.
            for i in range(rows):
                t += 1
                # Scale learning rate.
                # Default method in `scikit-learn`.
                eta = self.eta / (t ** 0.25)
                delta = self._bias + (self._weights * X[i]).sum() - y[i]
                # Adjust bias.
                derivative = delta
                if self.penalty:
                    penalty = self.penalty.compute(self.alpha, self._bias)
                    derivative += penalty
                self._bias -= eta * derivative
                # Adjust weights.
                for j in range(cols):
                    derivative = delta * X[i, j]
                    if self.penalty:
                        penalty = self.penalty.compute(self.alpha, self._weights[j])
                        derivative += penalty
                    self._weights[j] -= eta * derivative
                loss += (delta ** 2) / 2.
            # Calculate average loss.
            loss /= rows
            if loss >= self._best_loss - self.tolerance:
                stumble += 1
                if self.verbose:
                    logger.info("Stumbling: {} out of {}".format(stumble, self.stumble))
            else:
                stumble = 0
            if loss < self._best_loss:
                self._best_loss = loss
            # Print debug information.
            if self.verbose:
                norm = np.linalg.norm(self._weights)
                logger.info("-- Epoch {}, norm: {}, bias: {}, T: {}, avg loss: {:.06}"
                            .format(e, norm, self._bias, t, loss))
            # Return if convergence is achieved.
            if stumble > self.stumble:
                if self.verbose:
                    logger.info("Convergence after {} epochs".format(e))
                    return self

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa
        """
        Predict using the linear model.
        """
        predictions: List[float] = []
        rows, cols = X.shape
        for i in range(rows):
            prediction = self._bias + (self._weights * X[i]).sum()
            predictions.append(prediction)
        predictions: np.ndarray = np.asarray(predictions)

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # noqa
        """
        R-squared score.
        """
        prediction = self.predict(X)
        residual = ((y - prediction) ** 2).sum()
        total = ((y - y.mean()) ** 2).sum()
        score = 1. - (residual / total)

        return score

    def error(self, X: np.ndarray, y: np.ndarray) -> float:  # noqa
        """
        Mean squared error.
        """
        prediction = self.predict(X)
        error = ((prediction - y) ** 2).sum()
        error = error / len(prediction)
        error = np.sqrt(error)

        return error

    @property
    def bias(self) -> float:
        if self._bias is None:
            raise AttributeError("`Fit` was not performed")
        return self._bias

    @bias.setter
    def bias(self, value: Any):
        raise AttributeError("Field `bias` is immutable")

    @property
    def weights(self) -> np.ndarray:
        """
        Weights assigned to the features.
        """
        if self._weights is None:
            raise AttributeError("`Fit` was not performed")
        return self._weights

    @weights.setter
    def weights(self, value: Any):
        raise AttributeError("Field `weights` is immutable")
