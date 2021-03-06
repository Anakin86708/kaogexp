from math import sqrt
from unittest import TestCase, mock

import pandas as pd

from kaogexp.data.loader import ColunaYSingleton
from kaogexp.explainer.kaog.custom_kaog import NovaDistancia
from util import Data


class TestDistancias_(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        ColunaYSingleton().NOME_COLUNA_Y = 'target'

    def setUp(self) -> None:
        self.adult = Data.create_new_instance_adult(5)

    @mock.patch('kaogexp.explainer.kaog.custom_kaog.NovaDistancia._calcular_distancias_e_vizinhos')
    def test__tratar_x(self, mock_vizinhos):
        mock_vizinhos.return_value = ([], [])

        adult = self.adult
        tratador = adult.tratador
        colunas_encoded_drop = tratador.nomes_colunas_encoded.drop(ColunaYSingleton().NOME_COLUNA_Y)
        colunas_encoded = tratador.nomes_colunas_categoricas_encoded
        data = adult.x(True, False).sample(100)

        instance = NovaDistancia(adult)
        result: pd.DataFrame = instance._tratar_x(data)

        self.assertTrue((result.columns == colunas_encoded_drop).all())
        self.assertFalse((result[colunas_encoded] == 1).any().any())
        self.assertTrue((result[colunas_encoded] == sqrt(.5)).any().any())
