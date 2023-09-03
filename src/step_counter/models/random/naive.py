"""Module contains naive and dummy classifiers, which can be used to generate baseline accuracies
or to test training and evaluation pipelines."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted


class RandomStepClassifier(BaseEstimator, ClassifierMixin):
    """
    This classifier randomly predicts classes and should only be used to compare
     other classifiers with random baseline.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X, y):
        """
        Fit classifier
        Args:
            X: array with features
            y: array with targets

        Returns:
            'Fitted' random classifier

        """

        X, y = check_X_y(X, y)
        self.random_state_ = np.random.RandomState(seed=self.seed)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """
        Predict on X using randomly generated values
        Args:
            X: array with features

        Returns:
            Random predictions with the shape (len(X), 1)

        """
        y_pred = self.predict_proba(X)
        class_idx = np.argmax(y_pred, axis=1)
        return np.array([self.classes_[idx] for idx in class_idx])

    def predict_proba(self, X):
        """
        Predict probabilities on X using randomly generated values
        Args:
            X: array with features

        Returns:
            Random predictions with the shape (len(X), n_classes)

        """
        check_is_fitted(self)
        X = check_array(X)
        input_features = X.shape[1]
        if self.n_features_in_ is not None and input_features != self.n_features_in_:
            raise ValueError(
                "input_features should have length equal to number of "
                f"features ({self.n_features_in_}), got {input_features}"
            )
        self.random_state_.seed(self.seed)
        random_result = self.random_state_.uniform(
            size=(X.shape[0], len(self.classes_))
        )
        # normalize to sum to 1
        normalized_random_result = random_result / np.sum(
            random_result, axis=1, keepdims=True
        )
        return normalized_random_result

    def _more_tags(self):
        return {"poor_score": True, "non_deterministic": True}
