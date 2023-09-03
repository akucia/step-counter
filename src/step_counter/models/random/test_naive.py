"""Tests for naive dummy classifiers"""
from sklearn.utils.estimator_checks import parametrize_with_checks

from step_counter.models.random.naive import RandomStepClassifier


@parametrize_with_checks([RandomStepClassifier()])
def test_sklearn_compatible_estimator(estimator, check):
    """Test if custom random classifier respects scikit-learn api standards"""
    check(estimator)
