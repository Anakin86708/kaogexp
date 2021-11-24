import unittest

import pandas as pd
from kaog import KAOG
from sklearn.datasets import load_iris

from kaogexp.explainer.methods.Counterfactual import Counterfactual


class CounterfactualTest(unittest.TestCase):

    def setUp(self) -> None:
        iris = load_iris()
        x = pd.DataFrame(iris.data, columns=iris.feature_names)
        x = x.drop(columns=['sepal width (cm)', 'petal length (cm)'])
        y = pd.DataFrame(iris.target, columns=['target'])
        data = pd.concat([x, y], axis=1)
        self.iris = data.drop_duplicates()

        self.kaog = KAOG(self.iris)

    def test_instance_counterfactual(self):
        buscado = self.iris.sample(1).iloc[0]
        instance = Counterfactual(self.kaog, buscado)
        self.assertIsInstance(instance, Counterfactual)

    def test_realizar_busca(self):
        buscado = self.iris.iloc[42]

    def test_obter_vizinhos(self):
        buscado = self.iris.sample(1).iloc[0]
        instance = Counterfactual(self.kaog, buscado)
        vizinhos = instance._obter_vizinhos()

        shape_ = self.iris.shape[0] - 1
        self.assertLessEqual(vizinhos.shape[0], shape_)

    def test_e_classe_diferente(self):
        buscado = self.iris.loc[0]
        instance = Counterfactual(self.kaog, buscado)

        self.assertTrue(instance._e_classe_diferente(140))
        self.assertFalse(instance._e_classe_diferente(1))

    def test_e_maior_pureza(self):
        buscado = self.iris.loc[106]
        instance = Counterfactual(self.kaog, buscado)

        self.assertTrue(instance._e_maior_pureza(59))

    def test_condicao_busca(self):
        buscado = self.iris.loc[106]
        instance = Counterfactual(self.kaog, buscado)

        self.assertTrue(instance._condicao_busca(59))


if __name__ == '__main__':
    unittest.main()
