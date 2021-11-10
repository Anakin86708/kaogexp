import unittest
from unittest import expectedFailure

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
        instance_and_indexes = [
            (self.iris_instance, pd.Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])),
            (self.adult_instance,
             pd.Index(['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])),
        ]
        for instance, index in instance_and_indexes:
            with self.subTest(f"{instance.__class__}", instance=instance, index=index):
                pd.testing.assert_index_equal(instance.nomes_colunas_numericas, index)

    def test_x_iris(self):
        instance = self.iris_instance
        shape = (150, 4)
        index = pd.Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

        self.assertTrue(instance.x().shape == shape)
        pd.testing.assert_index_equal(instance.x().columns, index)

    def test_x_adult(self):
        instance = self.adult_instance
        shape = (32561, 14)
        index = pd.Index(
            ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
             'relationship',
             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])

        self.assertTrue(instance.x().shape == shape)
        pd.testing.assert_index_equal(instance.x().columns, index)

    def test_y(self):
        instances_and_shapes = [
            (self.iris_instance, (150,)),
            (self.adult_instance, (32561,)),
        ]
        for instance, shape in instances_and_shapes:
            with self.subTest(f"{instance.__class__}", instance=instance, shape=shape):
                self.assertEqual(instance.y().name, 'target')
                self.assertTrue(instance.y().shape == shape)

    def test_normalization_dataset_iris(self):
        instance = self.iris_instance

        nomes_cols_normalizar = instance.normalizador.nomes_colunas_normalizar
        self.assertTrue(
            instance.dataset(True).apply(lambda x: x[nomes_cols_normalizar].between(0, 1).all(), axis=1).all()
        )

    def test_normalization_dataset_adult(self):
        instance = self.adult_instance

        nomes_cols_normalizar = instance.normalizador.nomes_colunas_normalizar
        self.assertTrue(
            instance.dataset(True).apply(lambda x: x[nomes_cols_normalizar].between(0, 1).all(), axis=1).all()
        )

    @expectedFailure
    def test_normalization_fail(self):
        instances = [self.create_new_instance_iris(), self.create_new_instance_adult()]
        for instance in instances:
            with self.subTest(f"{instance.__class__}", instance=instance):
                nomes_cols_normalizar = instance.normalizador.nomes_colunas_normalizar
                self.assertTrue(
                    instance.dataset(False).apply(lambda x: x[nomes_cols_normalizar].between(0, 1).all(), axis=1).all()
                )

    ################
    # Util methods #
    ################

    @property
    def adult_dataset(self) -> pd.DataFrame:
        return Data.adult_dataset()

    @property
    def iris_instance(self):
        return self.create_new_instance_iris()

    @property
    def adult_instance(self):
        return self.create_new_instance_adult()

    @staticmethod
    def create_new_instance_iris(tratar_na=True):
        colunas_categoricas, df = Data.iris_dataset()
        return DatasetFromMemory(df, colunas_categoricas, tratar_na=tratar_na)

    def create_new_instance_adult(self, tratar_na=True):
        df = self.adult_dataset

        df['target'] = pd.Categorical(df['target'])
        df['target'].replace([' <=50K', ' >50K'], [0, 1], inplace=True)
        object__columns = df.select_dtypes(['object']).columns
        df[object__columns] = df[object__columns].astype('category')
        colunas_categoricas = df.drop('target', axis=1, errors='ignore').select_dtypes(['category', 'object']).columns
        return DatasetFromMemory(df, colunas_categoricas, tratar_na=tratar_na)


if __name__ == '__main__':
    unittest.main()
