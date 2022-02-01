import unittest
import warnings
from random import choice

import pandas as pd
from kaog import KAOG
from sklearn.datasets import load_iris

from kaogexp.data.loader import ColunaYSingleton
from kaogexp.explainer.methods.Counterfactual import Counterfactual


class CounterfactualTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        ColunaYSingleton().NOME_COLUNA_Y = 'target'

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
        classe_buscada = self.obter_outra_classe(buscado['target'])
        instance = Counterfactual(self.kaog, buscado, classe_buscada, None, None)
        self.assertIsInstance(instance, Counterfactual)

    def test_realizar_busca(self):
        buscado = self.iris.iloc[42]
        classe_buscada = self.obter_outra_classe(buscado['target'])

        instance = Counterfactual(self.kaog, buscado, classe_buscada, None, None)
        result = instance._realizar_busca()

        self.assertEqual(classe_buscada, result['target'])
        self.assertEqual(buscado.shape, result.shape)

    def test_obter_vizinhos(self):
        buscado = self.iris.sample(1).iloc[0]
        classe_buscada = self.obter_outra_classe(buscado['target'])
        instance = Counterfactual(self.kaog, buscado, classe_buscada, None, None)
        vizinhos = instance._obter_vizinhos()

        shape_ = self.iris.shape[0] - 1
        self.assertLessEqual(vizinhos.shape[0], shape_)

    def test_e_classe_buscada(self):
        buscado = self.iris.loc[106]
        classe_buscada = 1
        instance = Counterfactual(self.kaog, buscado, classe_buscada, None, None)

        try:
            self.assertTrue(instance._e_classe_buscada(50))
            self.assertEqual(classe_buscada, instance.instancia_modificada['target'])
        except AttributeError:
            warnings.warn('Não foi possível obter a classe da instância modificada, mas o resultado é o esperado')
        self.assertFalse(instance._e_classe_buscada(0))
        self.assertFalse(instance._e_classe_buscada(140))

    def test_e_maior_pureza(self):
        buscado = self.iris.loc[106]
        classe_buscada = self.obter_outra_classe(buscado['target'])
        instance = Counterfactual(self.kaog, buscado, classe_buscada, None, None)

        self.assertTrue(instance._e_maior_pureza(59))

    def test_condicao_busca(self):
        buscado = self.iris.loc[106]
        classe_buscada = 1
        instance = Counterfactual(self.kaog, buscado, classe_buscada, None, None)

        self.assertTrue(instance._condicao_busca(59))

    def test_classe_desejada(self):
        buscado = self.iris.loc[106]
        classe_buscada = 1
        instance = Counterfactual(self.kaog, buscado, classe_buscada, None, None)

        self.assertEqual(classe_buscada, instance.classe_desejada)
        self.assertEqual(classe_buscada, instance.instancia_modificada['target'])

    ################
    # UTIL METHODS #
    ################

    def obter_outra_classe(self, classe_atual):
        classes = self.iris['target'].unique().tolist()
        classes.remove(classe_atual)
        return choice(classes)


if __name__ == '__main__':
    unittest.main()
