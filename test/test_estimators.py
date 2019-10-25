import unittest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from example import GenericPythonFunctionEstimator


class TestGenericPythonFunctionEstimator(unittest.TestCase):
    def setUp(self):
        self.estimator = GenericPythonFunctionEstimator()
        self.X = np.asarray([[1, 2], [3, 2], [5, 6]])

        self.labels = ["shoes", "shoes", "bags"]
        self.le = LabelEncoder()
        self.le.fit(self.labels)
        self.y = self.le.transform(self.labels)

        self.test_predictors = np.asarray([[1, 2], [5, 6], [0, 0], [1000, 2000]])

        self.test_targets = ["shoes", "bags", "shoes", "bags"]

    def test_fit(self):
        self.estimator.fit(X=self.X, y=self.y)
        self.assertEqual(self.estimator.X_.shape, (3, 2))
        self.assertEqual(self.estimator.y_.shape, (3,))
        self.assertEqual(self.estimator.classes_.shape, (2,))

    def test_predict(self):
        self.estimator.fit(X=self.X, y=self.y)
        preds = self.estimator.predict(self.test_predictors)
        self.assertListEqual(self.le.classes_[preds].tolist(), self.test_targets)


if __name__ == '__main__':
    unittest.main()
