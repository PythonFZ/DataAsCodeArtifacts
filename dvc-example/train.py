import os
import pickle
import sys

import numpy as np
import package1
from sklearn.ensemble import RandomForestClassifier


def train(seed, n_est, min_split, matrix):
    """
    Train a random forest classifier.

    Args:
        seed (int): Random seed.
        n_est (int): Number of trees in the forest.
        min_split (int): Minimum number of samples required to split an internal node.
        matrix (scipy.sparse.csr_matrix): Input matrix.

    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained classifier.
    """
    labels = np.squeeze(matrix[:, 1].toarray())
    x = matrix[:, 2:]

    sys.stderr.write("Input matrix size {}\n".format(matrix.shape))
    sys.stderr.write("X matrix size {}\n".format(x.shape))
    sys.stderr.write("Y matrix size {}\n".format(labels.shape))

    clf = RandomForestClassifier(
        n_estimators=n_est, min_samples_split=min_split, n_jobs=2, random_state=seed
    )

    clf.fit(x, labels)

    return clf


class Train(package1.Node):
    """Train a ML model.
    
    Attributes
    ----------
    min_split : float, optional
        Minimum number of samples required to split an internal node.
    n_est : int, optional
        Number of trees in the forest.
    seed : int, optional
        Random seed.
    features : str, optional
        Path to the features directory.
    model : str, optional
        Path to the output model.
    """
    min_split: float = package1.params(0.01)
    n_est: int = package1.params(50)
    seed: int = package1.params(20170428)

    features: str = package1.deps_path("data/features")
    model: str = package1.outs_path("model.pkl")

    def run(self):
        with open(os.path.join(self.features, "train.pkl"), "rb") as fd:
            matrix, _ = pickle.load(fd)

        clf = train(
            seed=self.seed, n_est=self.n_est, min_split=self.min_split, matrix=matrix
        )

        # Save the model
        with open(self.model, "wb") as fd:
            pickle.dump(clf, fd)
