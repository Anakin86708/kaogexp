from unittest import TestCase

import pandas as pd

from data.sampler.categorical_sampler import RandomCategoricalSampler


class TestRandomCategoricalSampler(TestCase):

    def test__definir_colunas_alteradas(self):
        cat_cols = pd.Index(['a', 'b', 'c'])
        fixed_cols = pd.Index(['b'])
        expected = pd.Index(['a', 'c'])
        instance = RandomCategoricalSampler(pd.DataFrame(), cat_cols, fixed_cols)

        result = instance._definir_colunas_alteradas()

        print(result)
        pd.testing.assert_index_equal(expected, result)
