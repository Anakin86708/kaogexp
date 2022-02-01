import unittest

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from kaogexp.data.loader import ColunaYSingleton
from kaogexp.model.RandomForestModel import RandomForestModel
from util import Data


class RandomForestModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        ColunaYSingleton().NOME_COLUNA_Y = 'target'

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

    def test_create_from_dataset(self):
        dataset = Data.create_new_instance_adult()
        model = RandomForestModel.from_dataset(dataset)
        self.assertIsInstance(model, RandomForestModel)
        self.assertIsInstance(model.raw_model, RandomForestClassifier)

    ################
    # Util methods #
    ################

    def _prepare_data(self):
        data = Data.iris_dataset()
        target_col = 'target'
        x = data.drop(target_col, axis=1)
        y = data[target_col]
        model = RandomForestModel(x, y, None)
        return model, x


if __name__ == '__main__':
    unittest.main()
