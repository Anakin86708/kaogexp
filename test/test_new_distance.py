import unittest

import numpy as np
import pandas as pd
from kaog.util import NOME_COLUNA_Y

from main.new_distance import NewDistance
from util import Data


class NewDistanceTest(unittest.TestCase):

    def setUp(self) -> None:
        # IRIS
        # Result from Wolfram Alpha(euclidean distance [5.1, 3.5, 1.4, .2] and [4.9, 3, 1.4, .2])
        self.distance_iris = .538516
        self.iris_expected_x = np.array([5.1, 3.5, 1.4, .2])
        self.iris_expected_y = np.array([4.9, 3, 1.4, .2])

        # ADULT
        # Result from distance.euclidean com x e y normalizados e encoded
        # tem 4 categoricas diferrentes
        self.distance_adult = 2.02462
        self.adult_expected_x = np.array([0.30136986301369856, 'State-gov', 0.04430189755640374, 'Bachelors',
                                          0.8, 'Never-married', 'Adm-clerical', 'Not-in-family', 'White',
                                          'Male', 0.021740217402174022, 0.0, 0.39795918367346933,
                                          'United-States'], dtype='object')
        self.adult_expected_y = np.array([0.4520547945205479, 'Self-emp-not-inc', 0.04823759525135491,
                                          'Bachelors', 0.8, 'Married-civ-spouse', 'Exec-managerial',
                                          'Husband', 'White', 'Male', 0.0, 0.0, 0.12244897959183672,
                                          'United-States'], dtype='object')

    def test_distance_iris_as_series(self):
        expected, instance, x, y = self._arrange_iris()

        result = instance.calculate(x, y)

        np.testing.assert_array_equal(self.iris_expected_x, x.to_numpy())
        np.testing.assert_array_equal(self.iris_expected_y, y.to_numpy())
        self.assertAlmostEqual(expected, result, places=5)

    def test_distance_iris_as_array(self):
        expected, instance, x, y = self._arrange_iris()
        x = x.to_numpy()
        y = y.to_numpy()

        result = instance.calculate(x, y)

        np.testing.assert_array_equal(self.iris_expected_x, x)
        np.testing.assert_array_equal(self.iris_expected_y, y)
        self.assertAlmostEqual(expected, result, places=5)

    def test_distance_adult(self):
        instance_adult = Data.create_new_instance_adult()
        data = instance_adult.x(normalizado=True, encoded=False)
        x = data.iloc[0]
        y = data.loc[1]
        cat_cols = instance_adult.nomes_colunas_categoricas
        instance = NewDistance(data, cat_cols)
        expected = self.distance_adult

        result = instance.calculate(x, y)

        np.testing.assert_array_equal(self.adult_expected_x, x.to_numpy())
        np.testing.assert_array_equal(self.adult_expected_y, y.to_numpy())
        self.assertAlmostEqual(expected, result, places=4)

    def _arrange_iris(self):
        data = Data.iris_dataset().drop(columns=NOME_COLUNA_Y)
        x: pd.Series = data.loc[0]
        y: pd.Series = data.loc[1]
        instance = NewDistance(data, pd.Index([]))
        expected = self.distance_iris
        return expected, instance, x, y


if __name__ == '__main__':
    unittest.main()
