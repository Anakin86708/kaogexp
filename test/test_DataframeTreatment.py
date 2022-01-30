import unittest
from unittest import expectedFailure

import numpy as np
import pandas as pd

from data.loader import ColunaYSingleton
from kaogexp.data.treatment.DatasetTreatment import DatasetTreatment
from test.util import Data


class DatasetTreatmentTest(unittest.TestCase):

    def test_tratar_na_iris(self):
        dataset = Data.iris_dataset()

        # Colocar alguns dados como NaN
        index_last = dataset.iloc[-1].name
        dataset.loc[0, 'sepal_length'] = "?"
        dataset.loc[1, 'sepal_length'] = None
        dataset.loc[2, 'petal_length'] = None
        dataset.loc[4, 'sepal_length'] = "?"
        dataset.loc[4, 'sepal_width'] = "?"
        dataset.loc[4, 'petal_length'] = "?"
        dataset.loc[4, 'petal_width'] = "?"
        dataset.loc[146, 'sepal_width'] = "?"
        dataset.loc[146, 'sepal_length'] = None
        dataset.loc[index_last, 'sepal_width'] = "?"

        # Tratar os dados
        tratado = DatasetTreatment.tratar_na(dataset, (None, "?"))
        self.assertTrue(not tratado.isna().any().any())
        self.assertEqual(dataset.shape[0], tratado.shape[0])

    def test_tratar_na_adult(self):
        dataset = self.adult_dataset.copy()

        # Colocar alguns dados como NaN
        index_last = dataset.iloc[-1].name
        dataset.loc[0, 'age'] = None
        dataset.loc[0, 'fnlwgt'] = "?"
        dataset.loc[0, 'hours-per-week'] = "?"
        dataset.loc[0, 'workclass'] = "?"
        dataset.loc[0, 'education'] = "?"
        dataset.loc[0, 'marital-status'] = "?"
        dataset.loc[0, 'occupation'] = "?"
        dataset.loc[0, 'relationship'] = "?"
        dataset.loc[146, 'sex'] = "?"
        dataset.loc[index_last, 'capital-loss'] = "?"
        dataset.loc[index_last, 'capital-loss'] = "?"
        dataset.loc[index_last, 'education-num'] = "?"

        # Definir colunas categoricas
        colunas_categoricas = dataset.select_dtypes(include=['object']).columns
        dataset[colunas_categoricas] = dataset[colunas_categoricas].astype("category")
        assert DatasetTreatment._obter_cols_categoricas(dataset).size > 0

        # Tratar os dados
        tratado = DatasetTreatment.tratar_na(dataset, (None, "?"))
        self.assertTrue(not tratado.isna().any().any())
        self.assertEqual(dataset.shape[0], tratado.shape[0])

    def test_encode_iris_dataset(self):
        dataset, instance = self._prepare_iris()
        # Como o iris não possui colunas categóricas, o dataset de retorno deve ser o mesmo
        encode = instance.encode(dataset)
        self.assertEqual(dataset.shape, encode.shape)

    def test_encode_iris_series(self):
        dataset, instance = self._prepare_iris()
        input_ = dataset.sample(1).iloc[0]
        # Como o iris não possui colunas categóricas, o dataset de retorno deve ser o mesmo
        self.assertEqual(input_.shape, instance.encode(input_).shape)

    def test_encode_adult_dataset(self):
        dataset, instance = self._prepare_adult()

        # Tratar os dados
        encoded = instance.encode(dataset)
        pd.testing.assert_index_equal(instance.nomes_colunas_encoded, encoded.columns, check_order=False)
        self.assertEqual(0,
                         encoded.drop(ColunaYSingleton().NOME_COLUNA_Y, axis=1).select_dtypes(["O", "category"]).shape[
                             1])
        self.assertIn(ColunaYSingleton().NOME_COLUNA_Y, encoded.columns)

    def test_encode_adult_series(self):
        dataset, instance = self._prepare_adult()

        # Tratar os dados
        input_ = dataset.sample(1).iloc[0]
        encoded = instance.encode(input_)
        pd.testing.assert_index_equal(instance.nomes_colunas_encoded, encoded.index, check_order=False)
        # self.assertTrue(encoded.dtypes != "O" or encoded.dtypes != "category")

    def test_encode_iris_using_dataset(self):
        dataset = Data.create_new_instance_iris()
        encoded = dataset.dataset(encoded=True)

        self.assertEqual((150, 5), Data.iris_dataset().shape)

    def test_encode_iris_using_dataset_not_encoded(self):
        dataset = Data.create_new_instance_iris()
        encoded = dataset.dataset(encoded=False)

        self.assertEqual((150, 5), Data.iris_dataset().shape)

    def test_encode_adult_using_dataset(self):
        dataset = Data.create_new_instance_adult()
        encoded = dataset.dataset(encoded=True)

        pd.testing.assert_index_equal(dataset.tratador.nomes_colunas_encoded, encoded.columns, check_order=False)
        self.assertIn(ColunaYSingleton().NOME_COLUNA_Y, encoded.columns)
        self.assertEqual(0, encoded.select_dtypes(["O", "category"]).shape[1])

    @expectedFailure
    def test_encode_adult_using_dataset_fail(self):
        dataset = Data.create_new_instance_adult()
        encoded = dataset.dataset(encoded=False)

        pd.testing.assert_index_equal(dataset.tratador.nomes_colunas_encoded, encoded.columns, check_order=False)
        self.assertIn(ColunaYSingleton().NOME_COLUNA_Y, encoded.columns)
        self.assertEqual(0, encoded.select_dtypes(["O", "category"]).shape[1])

    def test_encode_expected_result(self):
        """Voltado para testar se as variáveis estão sendo codificadas corretamente"""
        dataset = pd.DataFrame(
            {
                "age": [25, 30, 24, 64],
                "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov"],
                "fnlwgt": [3500, 4500, 4500, 4500],
                "education": ["Bachelors", "HS-grad", "11th", "HS-grad"],
                "target": [0, 0, 1, 1]
            }
        )
        Data.define_categorical_from_object(dataset)

        expected_encoded = pd.DataFrame(
            {
                "age": [25, 30, 24, 64],
                "fnlwgt": [3500, 4500, 4500, 4500],
                "workclass_Federal-gov": [0, 0, 0, 1],
                "workclass_Private": [1, 0, 0, 0],
                "workclass_Self-emp-inc": [0, 0, 1, 0],
                "workclass_Self-emp-not-inc": [0, 1, 0, 0],
                "education_11th": [0, 0, 1, 0],
                "education_Bachelors": [1, 0, 0, 0],
                "education_HS-grad": [0, 1, 0, 1],
                "target": [0, 0, 1, 1],
            }
        )
        expected_columns = expected_encoded.columns

        instance = DatasetTreatment(dataset)
        encoded = instance.encode(dataset)
        pd.testing.assert_frame_equal(expected_encoded, encoded, check_dtype=False)
        pd.testing.assert_index_equal(expected_columns, encoded.columns, check_order=False)

    ################
    # Util methods #
    ################

    @property
    def adult_dataset(self) -> pd.DataFrame:
        return Data.adult_dataset()

    @property
    def iris_instance(self):
        return Data.create_new_instance_iris()

    @property
    def adult_instance(self):
        return Data.create_new_instance_adult()

    @staticmethod
    def _prepare_iris():
        dataset = Data.iris_dataset()
        instance = DatasetTreatment(dataset)
        # Coluna y deve ser ignorada
        dataset = dataset.drop(ColunaYSingleton().NOME_COLUNA_Y, axis=1)
        return dataset, instance

    def _prepare_adult(self, with_y=True):
        dataset = self.adult_dataset.copy()
        instance = DatasetTreatment(dataset)
        if not with_y:
            # Coluna y deve ser ignorada
            dataset = dataset.drop(ColunaYSingleton().NOME_COLUNA_Y, axis=1)
        # Remover valores faltantes
        dataset = dataset.replace(("?",), np.nan).dropna()
        # Definir colunas categoricas
        colunas_categoricas = dataset.select_dtypes(include=['object']).columns
        dataset[colunas_categoricas] = dataset[colunas_categoricas].astype("category")
        assert DatasetTreatment._obter_cols_categoricas(dataset).size > 0
        return dataset, instance


if __name__ == '__main__':
    unittest.main()
