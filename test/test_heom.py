import unittest

import numpy as np
import pandas as pd

from kaogexp.data.loader import ColunaYSingleton
from main.heom import MyHEOM
from util import Data


class HEOMTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        ColunaYSingleton().NOME_COLUNA_Y = 'target'

    def setUp(self) -> None:
        self.adult = Data.create_new_instance_adult()
        self.data = self.adult.dataset(True, False)
        self.categoricos = self.adult.nomes_colunas_categoricas

    def test_instance(self):
        instance = MyHEOM(self.data, self.categoricos)
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, MyHEOM)

    def test_distances(self):
        data = self.data.copy().drop(ColunaYSingleton().NOME_COLUNA_Y, axis=1)
        instance = MyHEOM(data, self.categoricos)
        num_cols = self.adult.nomes_colunas_numericas

        data_1: pd.Series = data.iloc[0]
        data_2: pd.Series = data.iloc[1]

        heom_distance = instance.heom(data_1, data_2)
        cat_distance = self.check_distances_cat(data_1, data_2)
        num_distance = self.get_distances_num(data_1, data_2, num_cols)

    def check_distances_cat(self, x: pd.Series, y: pd.Series):
        """Calcula a distância entre os registros de categóricos, sendo a soma, considerando 1 quando são iguais."""
        assert all(x.index == y.index)
        return x[self.categoricos] == y[self.categoricos]

    def get_distances_num(self, x: pd.Series, y: pd.Series, num_cols):
        range_ = np.max(self.data[num_cols]) - np.min(self.data[num_cols])
        return np.abs(x[num_cols] - y[num_cols]) / range_


if __name__ == '__main__':
    unittest.main()
