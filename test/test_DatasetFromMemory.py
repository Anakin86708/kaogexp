import unittest

import pandas as pd

from kaogexp.data.loader.DatasetFromMemory import DatasetFromMemory
from test.util import Data


class DatasetFromMemoryTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def test_create_instance(self):
        self.create_new_instance_iris()
        self.create_new_instance_adult()

    def test_categorical_columns(self):
        # Iris
        self.assertTrue(len(self.iris_instance.nomes_colunas_categoricas) == 0)

        # Adult
        cat_cols = ['workclass', 'education', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex',
                    'native-country']
        difference = self.adult_instance.nomes_colunas_categoricas.difference(cat_cols)
        self.assertTrue(len(difference) == 0, f"No match between categorical columns at adult dataset: {difference}")

    def test_numerical_columns(self):
        # Iris
        index = pd.Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        pd.testing.assert_index_equal(self.iris_instance.nomes_colunas_numericas, index)

        # Adult
        index = pd.Index(['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])
        pd.testing.assert_index_equal(self.adult_instance.nomes_colunas_numericas, index)

    def test_y_column(self):
        # Iris
        self.assertEqual(self.iris_instance.y().name, 'target')
        self.assertTrue(self.iris_instance.y().shape == (150,))

        # Adult
        self.assertEqual(self.adult_instance.y().name, 'target')
        self.assertTrue(self.adult_instance.y().shape == (32561,))

    ################
    # Util methods #
    ################

    @property
    def adult_dataset(self) -> pd.DataFrame:
        return Data.adult_dataset()

    @property
    def iris_instance(self):
        if not hasattr(self, '_iris_instance'):
            self._iris_instance = self.create_new_instance_iris()
        return self._iris_instance

    @property
    def adult_instance(self):
        if not hasattr(self, '_adult_instance'):
            self._adult_instance = self.create_new_instance_adult()
        return self._adult_instance

    @staticmethod
    def create_new_instance_iris():
        colunas_categoricas, df = Data.iris_dataset()
        return DatasetFromMemory(df, colunas_categoricas)

    def create_new_instance_adult(self):
        df = self.adult_dataset

        df['target'] = pd.Categorical(df['target'])
        df['target'].replace([' <=50K', ' >50K'], [0, 1], inplace=True)
        object__columns = df.select_dtypes(['object']).columns
        df[object__columns] = df[object__columns].astype('category')
        colunas_categoricas = df.drop('target', axis=1, errors='ignore').select_dtypes(['category', 'object']).columns
        return DatasetFromMemory(df, colunas_categoricas)


if __name__ == '__main__':
    unittest.main()
