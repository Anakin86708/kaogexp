import unittest
from math import sqrt

import pandas as pd

from kaogexp.metrics.proximity import Proximity
from main.new_distance import NewDistance


class FakeCounterfactual:
    def __init__(self, original, modificada):
        self.instancia_original = original
        self.instancia_modificada = modificada


class ProximityTest(unittest.TestCase):
    def test_numeric(self):
        original = pd.Series({'a': 1, 'b': 2, 'c': 3})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': 6})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = sqrt(2)

        dist = NewDistance(data, pd.Index([]))
        prox = Proximity(dist.calculate)
        result = prox.calcular(counterfactual)
        self.assertEqual(expected, result)

    def test_categorical(self):
        original = pd.Series({'a': 1, 'b': 3, 'c': "M"})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': "F"})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = sqrt(2)
        dist = NewDistance(data, pd.Index(['c']))
        prox = Proximity(dist.calculate)
        result = prox.calcular(counterfactual)
        self.assertEqual(expected, result)

    def test_categorical_same_num(self):
        original = pd.Series({'a': 1, 'b': 3, 'c': "M"})
        modificada = pd.Series({'a': 1, 'b': 3, 'c': "F"})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = 1
        dist = NewDistance(data, pd.Index(['c']))
        prox = Proximity(dist.calculate)
        result = prox.calcular(counterfactual)
        self.assertEqual(expected, result)

    def test_categorical_same_cat(self):
        original = pd.Series({'a': 1, 'b': 3, 'c': "F"})
        modificada = pd.Series({'a': 1, 'b': 5, 'c': "F"})
        data = pd.DataFrame([original, modificada])
        counterfactual = FakeCounterfactual(original, modificada)
        expected = 1
        dist = NewDistance(data, pd.Index(['c']))
        prox = Proximity(dist.calculate)
        result = prox.calcular(counterfactual)
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
