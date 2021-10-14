# Built-in.
import logging

# External.
import pandas as pd

# Internal.
from .crossval import CrossValidation
from .pipeline import Pipeline
from .preproc import StandardScaler
from .sgd import SGDRegressor

logging.basicConfig(level=logging.DEBUG)


def rename_features(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        0: "popularity",  # Popularity or support for the source of the document,
        1: "checkins",  # How many individuals so far visited this place,
        2: "interest",  # Daily interest of individuals towards source of the post,
        3: "category",
        # Category of the source of the document, e.g. place, institution, brand,
        29: "comments_before",  # Total number of comments before selected base datetime,
        30: "comments_last_24h",
        # Number of comments in last 24 hours, relative to base datetime,
        31: "comments_last_48h",
        # Number of comments in last 48 to last 24 hours relative to base datetime,
        32: "comments_first_24h",
        # Number of comments in the first 24 hours after the publication of post but before base datetime,
        34: "time",  # Selected time in order to simulate the scenario,
        35: "length",  # Character length of the post,
        36: "shares",  # No. of shares of the post,
        37: "is_promoted",  # Whether the post is promoted,
        38: "hours",  # `H` hours, for which we have comments received,
        39: "published_monday",
        40: "published_tuesday",
        41: "published_wednesday",
        42: "published_thursday",
        43: "published_friday",
        44: "published_saturday",
        45: "published_sunday",
        46: "current_monday",
        47: "current_tuesday",
        48: "current_wednesday",
        49: "current_thursday",
        50: "current_friday",
        51: "current_saturday",
        52: "current_sunday",
        53: "comments",  # Target: no. of comments in next `H` hrs.
    }
    df = df.rename(columns=columns)
    df["category"] = df["category"].astype("category")

    return df


def main():
    # Read data.
    paths = [
        f"./facebook/data/train/{i:03d}.csv"
        for i in range(1, 6)
    ]
    train = pd.concat([
        pd.read_csv(path, header=None)
        for path in paths
    ])
    train = rename_features(train)
    paths = [
        f"./facebook/data/test/{i:03d}.csv"
        for i in range(1, 10)
    ]
    test = pd.concat([
        pd.read_csv(path, header=None)
        for path in paths
    ])
    test = rename_features(test)
    # Learn.
    pipe = Pipeline([
        StandardScaler(),
        SGDRegressor(iterations=10 ** 5, stumble=24, tolerance=1e-3),
    ])
    cv = CrossValidation(folds=5)
    X, y = train.loc[:, train.columns != "comments"], train["comments"]
    for i_train, i_test in cv.split(X, y):
        X_train, y_train = X[i_train], y[i_train]
        X_test, y_test = X[i_test], y[i_test]
        pipe.fit(X_train, y_train)
        # Results.
        error = pipe.error(X_train, y_train)
        logging.info("RMSE on train: {}".format(error))
        error = pipe.error(X_test, y_test)
        logging.info("RMSE on test: {}".format(error))


if __name__ == "__main__":
    main()
