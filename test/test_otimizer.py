from unittest import TestCase
from unittest.mock import Mock, patch

import pandas as pd

from data.loader import NOME_COLUNA_Y
from data.loader.DatasetFromMemory import DatasetFromMemory
from data.sampler.LatinSampler import LatinSampler
from explainer.KAOGExp import KAOGExp
from explainer.methods.Counterfactual import Counterfactual
from explainer.otimizer import SparsityOptimization
from model.RandomForestModel import RandomForestModel
from test_KAOGExp import KAOGExpTest
from util import Data


class TestSparsityOptimization(TestCase):
    def test_optimize(self):
        adult = Data.create_new_instance_adult()
        data = adult.dataset(encoded=False)
        input_ = data[data[NOME_COLUNA_Y] == 0].sample(1).iloc[0]
        metodo = Counterfactual
        input_index = input_.name
        train_data = data.drop(input_index)
        fixed_cols = pd.Index(['age', 'sex', 'hours-per-week'])
        modelo = RandomForestModel(adult.x(encoded=True).drop(input_index), adult.y().drop(input_index), adult.tratador)
        sampler = LatinSampler(KAOGExpTest.EPSILON)
        kaogexp = KAOGExp(DatasetFromMemory(train_data, adult._nomes_colunas_categoricas), modelo, sampler,
                          fixed_cols=fixed_cols)

        classe_desejada = 1
        counterfactual: Counterfactual = kaogexp.explicar(input_, metodo, classe_desejada=classe_desejada,
                                                          normalizador_associado=adult.normalizador,
                                                          tratador_associado=adult.tratador)

        instance = SparsityOptimization(modelo, adult.nomes_colunas_categoricas)
        resultado: Counterfactual = instance.optimize(counterfactual)

        print(resultado)
        self.assertEqual(counterfactual.classe_desejada, resultado.classe_desejada)
        self.assertEqual(resultado.instancia_modificada[NOME_COLUNA_Y], resultado.classe_desejada)
        self.assertEqual(classe_desejada, modelo.predict(resultado.instancia_modificada))

    @patch('metrics.dispersao.Dispersao.calcular')
    def test__permutar_features(self, mock_dispersao):
        original = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        modificada = pd.Series([1, 5, 6, 7], index=['a', 'b', 'c', 'd'])
        mock_counterfactual = Mock()
        mock_counterfactual.instancia_original = original
        mock_counterfactual.instancia_modificada = modificada
        num_diff = 3
        mock_dispersao.return_value = num_diff

        instancia = SparsityOptimization(None, pd.Index([]))
        instancia._instancia_original = original.copy()
        resultado = pd.DataFrame(list(instancia._permutar_features(mock_counterfactual)))

        print(resultado)
        self.assertEqual((2 ** num_diff) - 2, len(resultado))
        self.assertTrue((resultado['a'] == 1).all())
