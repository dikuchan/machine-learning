# Built-in.
from typing import (
    Any, List, Optional,
)

# External.
import numpy as np


class Pipeline:
    def __init__(
            self,
            steps: List[Any],
    ):
        assert len(steps) > 0
        self._steps = steps[:-1]
        self._final = steps[-1]
        self._model: Optional[Any] = None

    def fit(
            self,
            X: np.ndarray,  # noqa
            y: Optional[np.ndarray] = None
    ) -> "Pipeline":
        for step in self._steps:
            model = step.fit(X, y)
            X, y = model.transform(X, y)  # noqa
        self._model = self._final.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._final.predict(X)

    def error(self, X: np.ndarray, y: np.ndarray) -> float:
        return self._final.error(X, y)
