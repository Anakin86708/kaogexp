import unittest
from math import sqrt
from unittest import expectedFailure

import pandas as pd

from kaogexp.data.loader import NOME_COLUNA_Y
from kaogexp.metrics.CERScore import CERScore
from kaogexp.metrics.dispersao import Dispersao
from kaogexp.metrics.proximity import Proximity
from kaogexp.metrics.validity import Validity
from main.new_distance import NewDistance


class FakeCounterfactual:
    def __init__(self, original, modificada, desejada=None):
        self.instancia_original = original
        self.instancia_modificada = modificada
        self.classe_desejada = desejada

    @property
    def classe_modificada(self):
        return self.instancia_modificada[NOME_COLUNA_Y]


class ProximityTest(unittest.TestCase):
    def test_numeric(self):
        original = pd.Series({'a': 1, 'b': 2, 'c': 3, NOME_COLUNA_Y: 1})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': 6, NOME_COLUNA_Y: 1})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = sqrt(2)

        dist = NewDistance(data, pd.Index([]))
        prox = Proximity(dist.calculate)
        result = prox.calcular(counterfactual)
        self.assertEqual(expected, result)

    def test_categorical(self):
        original = pd.Series({'a': 1, 'b': 3, 'c': "M", NOME_COLUNA_Y: 1})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': "F", NOME_COLUNA_Y: 1})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = sqrt(2)
        dist = NewDistance(data, pd.Index(['c']))
        prox = Proximity(dist.calculate)
        result = prox.calcular(counterfactual)
        self.assertEqual(expected, result)

    def test_categorical_same_num(self):
        original = pd.Series({'a': 1, 'b': 3, 'c': "M", NOME_COLUNA_Y: 1})
        modificada = pd.Series({'a': 1, 'b': 3, 'c': "F", NOME_COLUNA_Y: 1})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = 1
        dist = NewDistance(data, pd.Index(['c']))
        prox = Proximity(dist.calculate)
        result = prox.calcular(counterfactual)
        self.assertEqual(expected, result)

    def test_categorical_same_cat(self):
        original = pd.Series({'a': 1, 'b': 3, 'c': "F", NOME_COLUNA_Y: 1})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': "F", NOME_COLUNA_Y: 1})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = 1
        dist = NewDistance(data, pd.Index(['c']))
        prox = Proximity(dist.calculate)
        result = prox.calcular(counterfactual)
        self.assertEqual(expected, result)


class DispersionTest(unittest.TestCase):

    def test_numeric(self):
        original = pd.Series({'a': 1, 'b': 2, 'c': 3, 'target': 1})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': 6, 'target': 0})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = 2

        result = Dispersao.calcular(counterfactual)
        self.assertEqual(expected, result)

    def test_numeric_same_class(self):
        original = pd.Series({'a': 1, 'b': 2, 'c': 3, 'target': 1})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': 6, 'target': 1})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = 2

        result = Dispersao.calcular(counterfactual)
        self.assertEqual(expected, result)

    def test_numeric_none_changed(self):
        original = pd.Series({'a': 1, 'b': 2, 'c': 3, 'target': 1})
        modificada = pd.Series({'a': 1, 'b': 2, 'c': 3, 'target': 0})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = 0

        result = Dispersao.calcular(counterfactual)
        self.assertEqual(expected, result)

    def test_numeric_all_changed(self):
        original = pd.Series({'a': 1, 'b': 2, 'c': 3, 'target': 1})
        modificada = pd.Series({'a': 4, 'b': 5, 'c': 1, 'target': 0})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = 3

        result = Dispersao.calcular(counterfactual)
        self.assertEqual(expected, result)

    def test_categorical(self):
        original = pd.Series({'a': 1, 'b': 3, 'c': "M", 'target': 1})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': "F", 'target': 0})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = 2

        result = Dispersao.calcular(counterfactual)
        self.assertEqual(expected, result)

    def test_categorical_same_num(self):
        original = pd.Series({'a': 1, 'b': 3, 'c': "M", 'target': 1})
        modificada = pd.Series({'a': 1, 'b': 3, 'c': "F", 'target': 0})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = 1

        result = Dispersao.calcular(counterfactual)
        self.assertEqual(expected, result)

    def test_categorical_same_cat(self):
        original = pd.Series({'a': 1, 'b': 3, 'c': "F", 'target': 1})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': "F", 'target': 0})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = 1

        result = Dispersao.calcular(counterfactual)
        self.assertEqual(expected, result)


class ValidityTest(unittest.TestCase):

    def test_false_numeric(self):
        original = pd.Series({'a': 1, 'b': 2, 'c': 3, 'target': 1})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': 6, 'target': 0})
        counterfactual = FakeCounterfactual(original, modificada, 2)

        result = Validity.calcular(counterfactual)
        self.assertFalse(result)

    def test_true_numeric(self):
        original = pd.Series({'a': 1, 'b': 2, 'c': 3, 'target': 1})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': 6, 'target': 0})
        counterfactual = FakeCounterfactual(original, modificada, 0)

        result = Validity.calcular(counterfactual)
        self.assertTrue(result)

    @expectedFailure
    def test_numeric_fail(self):
        original = pd.Series({'a': 1, 'b': 2, 'c': 3, 'target': 1})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': 6, 'target': 0})
        counterfactual = FakeCounterfactual(original, modificada, 2)

        result = Validity.calcular(counterfactual)
        self.assertTrue(result)

    def test_false_categorical(self):
        original = pd.Series({'a': 1, 'b': 3, 'c': "M", 'target': 1})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': "F", 'target': 0})
        counterfactual = FakeCounterfactual(original, modificada, 2)

        result = Validity.calcular(counterfactual)
        self.assertFalse(result)

    def test_true_categorical(self):
        original = pd.Series({'a': 1, 'b': 3, 'c': "M", 'target': 1})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': "F", 'target': 0})
        counterfactual = FakeCounterfactual(original, modificada, 0)

        result = Validity.calcular(counterfactual)
        self.assertTrue(result)


class CERScoreTest(unittest.TestCase):

    def test_numeric(self):
        c1 = FakeCounterfactual(
            pd.Series({'a': 1, 'b': 2, 'c': 3, 'target': 1}),
            pd.Series({'a': 1, 'b': 5, 'c': 6, 'target': 0})
        )
        dist_c1 = sqrt(2)
        c2 = FakeCounterfactual(
            pd.Series({'a': 0, 'b': 1, 'c': 1, 'target': 0}),
            pd.Series({'a': 0, 'b': 1, 'c': 1, 'target': 0})
        )
        dist_c2 = 0
        c3 = FakeCounterfactual(
            pd.Series({'a': 1, 'b': 2, 'c': 3, 'target': 1}),
            pd.Series({'a': 1, 'b': 5, 'c': 6, 'target': 0})
        )
        dist_c3 = sqrt(2)
        counterfactuals = [c1, c2, c3]
        expected = .92436

        data = pd.DataFrame([
            pd.Series({'a': 1, 'b': 2, 'c': 3, 'target': 1}),
            pd.Series({'a': 1, 'b': 5, 'c': 6, 'target': 0}),
            pd.Series({'a': 0, 'b': 1, 'c': 1, 'target': 0}),
            pd.Series({'a': 0, 'b': 1, 'c': 1, 'target': 0}),
            pd.Series({'a': 1, 'b': 2, 'c': 3, 'target': 1}),
            pd.Series({'a': 1, 'b': 5, 'c': 6, 'target': 0})
        ])
        dist = NewDistance(data, pd.Index([]))
        cers = CERScore(dist.calculate)
        result = cers.calcular(counterfactuals)
        self.assertAlmostEqual(expected, result, places=5)


if __name__ == '__main__':
    unittest.main()
