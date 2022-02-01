from unittest import TestCase

import pandas as pd
from torch.jit import RecursiveScriptModule

from kaogexp.data.loader import ColunaYSingleton
from kaogexp.data.loader.DatasetFromMemory import DatasetFromMemory
from kaogexp.model.ANN import ANN


class TestANN(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        ColunaYSingleton().NOME_COLUNA_Y = 'income'

    def test__retrieve_model(self):
        result = ANN._retrieve_model('adult')

        self.assertIsInstance(result, RecursiveScriptModule)

    def test_predict_one_series(self):
        dataset = self._get_dateset()
        item = dataset.dataset(True, True).iloc[1]
        x = item.drop(ColunaYSingleton.NOME_COLUNA_Y)
        instance = ANN(dataset.tratador)
        expected = item[ColunaYSingleton.NOME_COLUNA_Y]

        result = instance.predict(x)

        self.assertEqual(expected, result)

    def test_predict_zero_series(self):
        dataset = self._get_dateset()
        item = dataset.dataset(True, True).iloc[2]
        x = item.drop(ColunaYSingleton.NOME_COLUNA_Y)
        instance = ANN(dataset.tratador)
        expected = item[ColunaYSingleton.NOME_COLUNA_Y]

        result = instance.predict(x)

        self.assertEqual(expected, result)

    def test_predict_one_dataframe(self):
        dataset = self._get_dateset()
        df = dataset.dataset(True, True)
        df = df[df['income'] == 1].iloc[:2]
        x = df.drop(ColunaYSingleton.NOME_COLUNA_Y, axis=1)
        instance = ANN(dataset.tratador)
        expected = df[ColunaYSingleton.NOME_COLUNA_Y]

        result = instance.predict(x)

        self.assertTrue((result == expected).all())

    def test_predict_zero_dataframe(self):
        dataset = self._get_dateset()
        df = dataset.dataset(True, True)
        df = df[df['income'] == 0].iloc[:2]
        x = df.drop(ColunaYSingleton.NOME_COLUNA_Y, axis=1)
        instance = ANN(dataset.tratador)
        expected = df[ColunaYSingleton.NOME_COLUNA_Y]

        result = instance.predict(x)

        self.assertTrue((result == expected).all())

    def _get_dateset(self):
        index = ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status', 'occupation', 'relationship',
                 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        data = pd.read_csv('../data/carla_data/adult_test.csv', index_col=0)
        data.columns = data.columns.str.strip()
        cat_index = pd.Index(['workclass', 'marital-status', 'occupation', 'relationship',
                              'race', 'sex', 'native-country'])
        ColunaYSingleton.NOME_COLUNA_Y = 'income'
        dataset = DatasetFromMemory(data, cat_index)
        return dataset

    @classmethod
    def tearDownClass(cls) -> None:
        ColunaYSingleton().NOME_COLUNA_Y = 'target'
