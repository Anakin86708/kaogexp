import random
import unittest
from copy import deepcopy
from functools import partial
from typing import Union
from unittest import expectedFailure

import numpy as np
import pandas as pd

from kaogexp.data.loader import NOME_COLUNA_Y
from kaogexp.data.sampler.LatinSampler import LatinSampler
from util import Data


class LatinSamplerTest(unittest.TestCase):
    SEED = 42
    EPSILON = 0.05

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        self.instance = self.create()

    def test_sanitize_numeric(self):
        # Sanitize np.array with only numeric values
        input_ = pd.Series([1, .432, 3])
        sanitized = self.instance._sanitize(input_)
        np.testing.assert_array_equal(sanitized, input_)
        self.assertTrue(sanitized.dtype, 'float')

    def test_sanitize_non_numeric(self):
        # Sanitize np.array with only non-numeric values
        input_ = pd.Series(['a', 'b', 'c'])
        expected = np.array([0] * len(input_))
        sanitized = self.instance._sanitize(input_)
        np.testing.assert_array_equal(sanitized, expected)
        self.assertTrue(sanitized.dtype, 'float')

    def test_sanitize_mixed(self):
        # Sanitize np.array with mixed numeric and non-numeric values
        input_ = pd.Series(['a', .432, 3])
        expected = np.array([0, .432, 3])
        sanitized = self.instance._sanitize(input_)
        np.testing.assert_array_equal(sanitized, expected)
        self.assertTrue(sanitized.dtype, 'float')

    def test_sanitize_real(self):
        # Sanitize a real series
        input_ = Data.adult_dataset().loc[0]
        as_zero = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'native-country', 'target']
        expected: pd.Series = deepcopy(input_)
        expected[as_zero] = 0
        sanitized = self.instance._sanitize(input_)
        pd.testing.assert_series_equal(expected, sanitized, check_dtype=False)

    def test_realizar_amostragem_origin(self):
        # Test with a numeric array on origin
        input_ = pd.Series(np.array([0, 0]))
        amount = 10
        sample = self.instance.realizar_amostragem(input_, amount)

        is_inside_space = partial(self.is_inside_space, epsilon=self.EPSILON, interest_point=input_)
        self.assertTrue(is_inside_space(sample).all(), "Sample is not inside the space")
        self.assertNotIn('object', sample.dtypes)

    def test_realizar_amostragem_multiple_points(self):
        # Test with a numeric array in multiple points
        num_subtests = 50
        min_shape_sample = 2
        max_shape_sample = 15

        for i in range(num_subtests):
            with self.subTest("Amostragem_multiple_points subtest", i=i):
                input_ = pd.Series(np.random.random_sample(random.randint(min_shape_sample, max_shape_sample)))
                amount = 10
                sample = self.instance.realizar_amostragem(input_, amount)

                is_inside_space = partial(self.is_inside_space, epsilon=self.EPSILON, interest_point=input_)
                self.assertTrue(is_inside_space(sample).all(), "Sample is not inside the space")
                self.assertNotIn('object', sample.dtypes)

    @expectedFailure
    def test_amostragem_diferentes(self):
        input_ = pd.Series(np.array([0, 0]))
        amount = 10
        sample1 = self.instance.realizar_amostragem(input_, amount).to_numpy()
        sample2 = self.instance.realizar_amostragem(input_, amount).to_numpy()
        # Comparar cada elemento de sample1 com cada elemento de sample2
        for i1, i2 in zip(sample1, sample2):
            with self.subTest("Amostragem_diferentes subtest", i1=i1, i2=i2):
                np.testing.assert_array_equal(i1, i2)

    def test_real_sampling(self):
        adult = Data.create_new_instance_adult()
        input_: pd.Series = adult.dataset(True, False).sample(1).iloc[0]
        amount = 10
        sample = self.instance.realizar_amostragem(input_, amount)
        colunas_numericas = adult.nomes_colunas_numericas
        colunas_catetgorias = adult.nomes_colunas_categoricas
        sample_num = sample[colunas_numericas]

        is_inside_space = partial(self.is_inside_space, epsilon=self.EPSILON, interest_point=input_[colunas_numericas])
        self.assertTrue(is_inside_space(sample_num).all(), "Sample is not inside the space")
        pd.testing.assert_index_equal(adult.tratador.nomes_colunas_originais.drop(NOME_COLUNA_Y), sample.columns)
        sample_ = sample.sample(1).iloc[0]
        pd.testing.assert_series_equal(input_[colunas_catetgorias], sample_[colunas_catetgorias], check_names=False)

    ################
    # Util methods #
    ################

    @staticmethod
    def create(epsilon=EPSILON, seed=SEED):
        return LatinSampler(epsilon, seed)

    @staticmethod
    def is_inside_space(x: pd.DataFrame, epsilon: Union[int, float, np.ndarray],
                        interest_point: pd.Series) -> bool:
        inf = interest_point - epsilon
        sup = interest_point + epsilon
        return (inf <= x.min()).all() and (x.max() <= sup).all()


if __name__ == '__main__':
    unittest.main()
