from unittest import TestCase

import pandas as pd
from torch.jit import RecursiveScriptModule

from model.ANN import ANN


class TestANN(TestCase):
    def test__retrieve_model(self):
        result = ANN._retrieve_model()

        self.assertIsInstance(result, RecursiveScriptModule)

    def test_predict(self):
        index = ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status', 'occupation', 'relationship',
                 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        data = pd.Series(
            [39, 0, 1, 77516, 13, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 2174,
             0, 40, 1, 0]
        )
        expected = 0
        instance = ANN(None)

        result = instance.predict(data)

        self.assertEqual(expected, result)
