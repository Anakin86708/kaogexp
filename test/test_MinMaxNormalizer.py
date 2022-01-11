import unittest

import pandas as pd

from kaogexp.data.normalizer.MinMaxNormalizer import MinMaxNormalizer
from test.util import Data


class MinMaxNormalizerTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Numeric columns from adult dataset
        self.numeric_columns_adult = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                                      'hours-per-week']
        self.data_adult = self.adult_dataset.sample(100)

    def test_create_instance(self):
        self.create()

    def test_transform_series(self):
        instance = self.create()

        # Test with a series
        sample = self.data_adult.sample(1).iloc[0]
        transformed = instance.transform(sample)
        self.assertTrue(transformed[instance.nomes_colunas_normalizar].between(0, 1).all())
        self.assertIsInstance(transformed, pd.Series)

    def test_transform_dataframe(self):
        instance = self.create()

        # Test with a dataframe with a single row
        sample = self.data_adult.sample(1)
        transformed = instance.transform(sample)
        self.assertTrue(
            transformed.apply(lambda x: x[instance.nomes_colunas_normalizar].between(0, 1).all(), axis=1).all())
        self.assertIsInstance(transformed, pd.DataFrame)

        # Test with a dataframe with multiple rows¶
        sample = self.data_adult.sample(10)
        transformed = instance.transform(sample)
        self.assertTrue(
            transformed.apply(lambda x: x[instance.nomes_colunas_normalizar].between(0, 1).all(), axis=1).all())
        self.assertIsInstance(transformed, pd.DataFrame)

    def test_reverse_transform_series(self):
        instance = self.create()

        # Test with a series
        sample = self.data_adult.sample(1).iloc[0]
        transformed = instance.transform(sample)
        reversed_ = instance.inverse_transform(transformed)
        pd.testing.assert_series_equal(reversed_, sample)
        self.assertIsInstance(reversed_, pd.Series)

    def test_reverse_transform_dataframe(self):
        instance = self.create()

        # Test with a dataframe with a single row
        sample = self.data_adult.sample(1)
        transformed = instance.transform(sample)
        reversed_ = instance.inverse_transform(transformed)
        pd.testing.assert_frame_equal(reversed_, sample)
        self.assertIsInstance(reversed_, pd.DataFrame)

        # Test with a dataframe with multiple rows
        sample = self.data_adult.sample(10)
        transformed = instance.transform(sample)
        reversed_ = instance.inverse_transform(transformed)
        pd.testing.assert_frame_equal(reversed_, sample)
        self.assertIsInstance(reversed_, pd.DataFrame)

    ################
    # Util methods #
    ################

    @property
    def adult_dataset(self) -> pd.DataFrame:
        return Data.adult_dataset()

    def create(self):
        x = self.data_adult.drop('target', axis=1)
        return MinMaxNormalizer(x, self.data_adult[self.numeric_columns_adult].columns)


if __name__ == '__main__':
    unittest.main()
