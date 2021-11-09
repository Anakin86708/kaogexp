import unittest

import pandas as pd
from sklearn.datasets import load_iris

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
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = pd.Categorical.from_codes(data.target, data.target_names)
        colunas_categoricas = df.drop('target', axis=1, errors='ignore').select_dtypes(['category', 'object']).columns
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
