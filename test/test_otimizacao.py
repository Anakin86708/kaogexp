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
        esperado = [
            pd.Series({'a': 1, 'b': 2, 'c': 3}),
            pd.Series({'a': 1, 'b': 5, 'c': 3}),
            pd.Series({'a': 1, 'b': 2, 'c': 6}),
            pd.Series({'a': 1, 'b': 5, 'c': 6}),
        ]

        result = instance._permutar_features(modificada, [], ['a', 'b', 'c'])
        self.assertEqual(len(esperado), len(result))
        for item in esperado:
            self.assertIn(item, result)


if __name__ == '__main__':
    unittest.main()
