import unittest

import pandas as pd

from kaogexp.data.normalizer.MinMaxNormalizer import MinMaxNormalizer
from test.util import Data


class MinMaxNormalizerTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        # Numeric columns from adult dataset
        super().__init__(*args, **kwargs)

        self.numeric_columns_adult = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                                      'hours-per-week']
        self.data_adult = self.adult_dataset.sample(100)

    def create(self):
        x = self.data_adult.drop('target', axis=1)
        return MinMaxNormalizer(x, self.data_adult[self.numeric_columns_adult].columns)

    def test_create_instance(self):
        self.create()

    def test_transform_series(self):
        instance = self.create()

        # Test with a series
        sample = self.data_adult.sample(1).iloc[0]
        transformed = instance.normalizar(sample)
        self.assertTrue(transformed[instance.nomes_colunas_numericas].between(0, 1).all())
        self.assertIsInstance(transformed, pd.Series)

        # Test with a dataframe with a single row
        sample = self.data_adult.sample(1)
        transformed = instance.normalizar(sample)
        self.assertTrue(
            transformed.apply(lambda x: x[instance.nomes_colunas_numericas].between(0, 1).all(), axis=1).all())
        self.assertIsInstance(transformed, pd.DataFrame)

        # Test with a dataframe with multiple rows¶
        sample = self.data_adult.sample(10)
        transformed = instance.normalizar(sample)
        self.assertTrue(
            transformed.apply(lambda x: x[instance.nomes_colunas_numericas].between(0, 1).all(), axis=1).all())
        self.assertIsInstance(transformed, pd.DataFrame)

    @property
    def adult_dataset(self) -> pd.DataFrame:
        return Data.adult_dataset()


if __name__ == '__main__':
    unittest.main()
