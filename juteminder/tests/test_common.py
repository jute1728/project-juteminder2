import pytest

from sklearn.utils.estimator_checks import check_estimator

from juteminder import TemplateEstimator
from juteminder import TemplateClassifier
from juteminder import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
