import random
import unittest
from functools import partial
from unittest import expectedFailure

import numpy as np
import pandas as pd

from kaogexp.data.sampler.LatinSampler import LatinSampler


class LatinSamplerTest(unittest.TestCase):
    SEED = 42
    EPSILON = 0.05

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        self.instance = self.create()

    def test_sanitize_numeric(self):
        # Sanitize np.array with only numeric values
        input_ = np.array([1, .432, 3])
        sanitized = self.instance._sanitize(input_)
        np.testing.assert_array_equal(sanitized, input_)
        self.assertTrue(sanitized.dtype, 'float')

    def test_sanitize_non_numeric(self):
        # Sanitize np.array with only non-numeric values
        input_ = np.array(['a', 'b', 'c'])
        expected = np.array([0] * len(input_))
        sanitized = self.instance._sanitize(input_)
        np.testing.assert_array_equal(sanitized, expected)
        self.assertTrue(sanitized.dtype, 'float')

    def test_sanitize_mixed(self):
        # Sanitize np.array with mixed numeric and non-numeric values
        input_ = np.array(['a', .432, 3])
        expected = np.array([0, .432, 3])
        sanitized = self.instance._sanitize(input_)
        np.testing.assert_array_equal(sanitized, expected)
        self.assertTrue(sanitized.dtype, 'float')

    def test_realizar_amostragem_origin(self):
        # Test with a numeric array on origin
        input_ = pd.Series(np.array([0, 0]))
        amount = 10
        sample = self.instance.realizar_amostragem(input_, amount)

        is_inside_space = np.vectorize(partial(self.is_inside_space, epsilon=self.EPSILON, interest_point=input_))
        self.assertTrue(is_inside_space(sample).all(), "Sample is not inside the space")

    def test_realizar_amostragem_multiple_points(self):
        # Test with a numeric array in multiple points
        num_subtests = 50
        min_shape_sample = 1
        max_shape_sample = 15

        for i in range(num_subtests):
            with self.subTest("Amostragem_multiple_points subtest", i=i):
                input_ = pd.Series(np.random.random_sample(random.randint(min_shape_sample, max_shape_sample)))
                amount = 10
                sample = self.instance.realizar_amostragem(input_, amount)

                results = list(map(partial(self.is_inside_space, epsilon=self.EPSILON, interest_point=input_), sample))
                self.assertTrue(False not in results, f"[{i}] Samples are not inside the correct space")

    @expectedFailure
    def test_amostragem_diferentes(self):
        input_ = pd.Series(np.array([0, 0]))
        amount = 10
        sample1 = self.instance.realizar_amostragem(input_, amount)
        sample2 = self.instance.realizar_amostragem(input_, amount)
        # Comparar cada elemento de sample1 com cada elemento de sample2
        for i1, i2 in zip(sample1, sample2):
            with self.subTest("Amostragem_diferentes subtest", i1=i1, i2=i2):
                np.testing.assert_array_equal(i1, i2)

    ################
    # Util methods #
    ################

    @staticmethod
    def create(epsilon=EPSILON, seed=SEED):
        return LatinSampler(epsilon, seed)

    @staticmethod
    def is_inside_space(x: np.ndarray, epsilon, interest_point) -> bool:
        inf = interest_point - epsilon
        sup = interest_point + epsilon
        return (inf <= x).all() and (x <= sup).all()


if __name__ == '__main__':
    unittest.main()
