from unittest import TestCase
from unittest.mock import Mock, patch

import pandas as pd

from data.loader import NOME_COLUNA_Y
from explainer.methods.Counterfactual import Counterfactual
from explainer.otimizer import SparsityOptimization
from model.RandomForestModel import RandomForestModel
from util import Data


class TestSparsityOptimization(TestCase):
    def test_optimize(self):
        classe_atual = 0
        classe_desejada = 1
        adult = Data.create_new_instance_adult()
        data = adult.dataset(encoded=False)
        input_ = data[data[NOME_COLUNA_Y] == classe_atual].sample(1).iloc[0]
        metodo = Counterfactual
        input_index = input_.name
        train_data = data.drop(input_index)
        fixed_cols = pd.Index(['age', 'sex', 'hours-per-week'])
        modelo = RandomForestModel(adult.x(encoded=True).drop(input_index), adult.y().drop(input_index), adult.tratador)
        mock_counterfactual = Mock()
        mock_counterfactual.configure_mock(**{'instancia_original.return_value': input_})

    @patch('metrics.dispersao.Dispersao.calcular')
    def test__permutar_features(self, mock_dispersao):
        original = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        modificada = pd.Series([1, 5, 6, 7], index=['a', 'b', 'c', 'd'])
        mock_counterfactual = Mock()
        mock_counterfactual.instancia_original = original
        mock_counterfactual.instancia_modificada = modificada
        mock_dispersao.return_value = 3

        instancia = SparsityOptimization(None, pd.Index([]))
        instancia._instancia_original = original.copy()
        resultado = instancia._permutar_features(mock_counterfactual)

        print(resultado)
