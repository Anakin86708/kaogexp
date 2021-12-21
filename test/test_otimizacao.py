import unittest

import pandas as pd

from kaogexp.explainer.otimizer import SparsityOptimization


class OtimizacaoTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_permutar_features(self):
        instance = SparsityOptimization(None, [])
        original = pd.Series({'a': 1, 'b': 2, 'c': 3})
        instance._instancia_original = original.copy()
        modificada = pd.Series({'a': 1, 'b': 5, 'c': 6})
        esperado = pd.DataFrame([
            pd.Series({'a': 1, 'b': 5, 'c': 6}),
            pd.Series({'a': 1, 'b': 2, 'c': 6}),
            pd.Series({'a': 1, 'b': 2, 'c': 3}),
            pd.Series({'a': 1, 'b': 5, 'c': 3}),
        ])

        result = instance._permutar_features(modificada, [], ['a', 'b', 'c'])
        self.assertEqual(len(esperado), len(result))
        pd.testing.assert_frame_equal(esperado, result, check_dtype=False, check_names=False, check_like=True)


if __name__ == '__main__':
    unittest.main()
