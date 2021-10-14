# External.
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Internal.
from .crossval import CrossValidation
from .preproc import StandardScaler as Scaler
from .sgd import SGDRegressor as SGD


def test_sgd_smoke():
    X = np.array([
        [1, 2],
        [2, 4],
        [3, 6],
    ], dtype=float)
    y = np.array([3, 6, 9], dtype=float)
    regressor = SGD(stumble=12)
    regressor.fit(X, y)

    assert regressor.score(X, y) > 0.


def test_sgd_sklearn():
    n_samples, n_features = 200, 4
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)  # noqa
    y = rng.randn(n_samples)  # noqa

    # Try fit with scikit-learn.
    regressor = make_pipeline(
        StandardScaler(),
        SGDRegressor(max_iter=1000, tol=1e-3),
    )
    regressor.fit(X, y)
    sl = regressor.score(X, y)

    # Try fit with the implemented SGD.
    regressor = make_pipeline(
        Scaler(),
        SGD(iterations=1000, tolerance=1e-3),
    )
    regressor.fit(X, y)
    sr = regressor.score(X, y)

    assert abs(sl - sr) < 1e-1


def test_cross_validation():
    n_samples, n_features = 10, 2
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)  # noqa

    cv = CrossValidation(folds=5)
    previous = None
    for fold in cv.split(X):
        train, test = fold
        train, test = X[train], X[test]
        n, _ = train.shape
        m, _ = test.shape
        assert n == 8
        assert m == 2
        if previous is not None:
            assert not np.array_equal(train, previous)
        previous = train
