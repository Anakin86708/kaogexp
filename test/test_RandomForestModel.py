import unittest

import numpy as np

from kaogexp.model.RandomForestModel import RandomForestModel
from util import Data


class RandomForestModelTest(unittest.TestCase):

    def test_predict_dataset(self):
        model, x = self._prepare_data()

        expected_shape = (x.shape[0],)
        predict = model.predict(x)
        self.assertEqual(expected_shape, predict.shape)

    def test_predict_series(self):
        model, x = self._prepare_data()

        x = x.sample(n=1).iloc[0]
        predict = model.predict(x)
        self.assertIsInstance(predict, np.number)

    ################
    # Util methods #
    ################

    def _prepare_data(self):
        data = Data.iris_dataset()
        target_col = 'target'
        x = data.drop(target_col, axis=1)
        y = data[target_col]
        model = RandomForestModel(x, y)
        return model, x


if __name__ == '__main__':
    unittest.main()
