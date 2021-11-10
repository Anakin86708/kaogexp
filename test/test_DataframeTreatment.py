import unittest

import pandas as pd

from kaogexp.data.treatment.DatasetTreatment import DatasetTreatment
from test.util import Data


class DatasetTreatmentTest(unittest.TestCase):

    def test_tratar_na_iris(self):
        _, dataset = Data.iris_dataset()

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
        dataset = self.adult_dataset

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
        assert DatasetTreatment._obter_cols_categoricas_and_y(dataset).size > 0

        # Tratar os dados
        tratado = DatasetTreatment.tratar_na(dataset, (None, "?"))
        self.assertTrue(not tratado.isna().any().any())
        self.assertEqual(dataset.shape[0], tratado.shape[0])

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


if __name__ == '__main__':
    unittest.main()
