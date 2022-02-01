from random import choice
from unittest import TestCase

import pandas as pd

from kaogexp.data.loader import ColunaYSingleton
from kaogexp.data.sampler.categorical_sampler import RandomCategoricalSampler


class TestRandomCategoricalSampler(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        ColunaYSingleton().NOME_COLUNA_Y = 'target'

    def setUp(self) -> None:
        self.data = pd.DataFrame(
            {
                'a': [1, 2, 3, 5, 3],
                'b': ['m', 'm', 'f', 'f', 'f'],
                'c': [1, 3, 3, 2, 1],
                'd': [.4, .2, .5, .2, 1]
            }
        )

    def test__definir_colunas_alteradas(self):
        cat_cols = pd.Index(['a', 'b', 'c'])
        fixed_cols = pd.Index(['b'])
        expected = pd.Index(['a', 'c'])
        instance = RandomCategoricalSampler(self.data, cat_cols, fixed_cols)

        result = instance._definir_colunas_alteradas()

        print(result)
        pd.testing.assert_index_equal(expected, result)

    def test_obter_valores_cat_unicos(self):
        cat_cols = pd.Index(['a', 'b', 'c'])
        fixed_cols = pd.Index(['c', 'd'])
        expected = {
            'a': [1, 2, 3, 5],
            'b': ['m', 'f']
        }

        instance = RandomCategoricalSampler(self.data, cat_cols, fixed_cols)
        result = instance._obter_valores_cat_unicos()

        print(result)
        self.assertDictEqual(expected, result)

    def test_realizar_amostragem(self):
        cat_cols = pd.Index(['a', 'b', 'c'])
        fixed_cols = pd.Index(['c', 'd'])
        num_amostras = 10
        amostra = pd.DataFrame(
            {
                'a': [choice(range(1, 6))] * num_amostras,
                'b': [choice(['f', 'm'])] * num_amostras,
                'c': [choice(range(1, 4))] * num_amostras,
                'd': [0] * num_amostras
            }
        )
        print('Original')
        print(amostra)
        instance = RandomCategoricalSampler(self.data, cat_cols, fixed_cols)

        result = instance.realizar_amostragem(amostra)

        print('Resultado')
        print(result)
        self.assertFalse(result.equals(amostra))
        pd.testing.assert_frame_equal(amostra[fixed_cols], result[fixed_cols])
