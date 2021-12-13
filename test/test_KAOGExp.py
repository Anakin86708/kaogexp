import unittest
from unittest import expectedFailure

import pandas as pd
from kaog import KAOG

from kaogexp.data.loader import NOME_COLUNA_Y
from kaogexp.data.sampler.LatinSampler import LatinSampler
from kaogexp.explainer.KAOGExp import KAOGExp
from kaogexp.explainer.methods.Counterfactual import Counterfactual
from kaogexp.model.RandomForestModel import RandomForestModel
from util import Data


class KAOGExpTest(unittest.TestCase):
    EPSILON = 0.05
    QTD_AMOSTRAS = 10

    def setUp(self) -> None:
        self.adult = Data.create_new_instance_adult()
        model = RandomForestModel.from_dataset(self.adult)
        sampler = LatinSampler(KAOGExpTest.EPSILON)
        self.instance = KAOGExp(self.adult, model, sampler)

    def test_assert_instance_compatibility_dataframe(self):
        """Instance must be `pd.Series` or `pd.DataFrame` and with same columns as the `tratador`"""
        adult = Data.create_new_instance_adult()
        input_ = adult.dataset(encoded=False).sample(5)
        self.instance._assert_instance_compatibility(input_)

    def test_assert_instance_compatibility_series(self):
        """Instance must be `pd.Series` or `pd.DataFrame` and with same columns as the `tratador`"""
        adult = Data.create_new_instance_adult()
        input_ = adult.dataset(encoded=False).sample(5).iloc[0]
        self.instance._assert_instance_compatibility(input_)

    @expectedFailure
    def test_assert_instance_compatibility_error(self):
        """Instance must be `pd.Series` or `pd.DataFrame` and with same columns as the `tratador`, not `np.ndarray`"""
        adult = Data.create_new_instance_adult()
        input_ = adult.dataset(encoded=False).sample(5).iloc[0].to_numpy()
        self.instance._assert_instance_compatibility(input_)

    def test_realizar_amostragem(self):
        """Realizar amostragem must return a `pd.DataFrame` with the right amount, defined in the class field."""
        adult = Data.create_new_instance_adult()
        input_ = adult.dataset(encoded=False).sample(1).iloc[0]
        sample = self.instance._realizar_amostragem(input_)
        self.assertEqual(KAOGExp.NUM_SAMPLES, sample.shape[0])
        pd.testing.assert_index_equal(input_.index.drop(NOME_COLUNA_Y), sample.columns)

    def test_realizar_amostragem_dataframe_raise(self):
        """Realizar amostragem must recive `pd.Series` only."""
        adult = Data.create_new_instance_adult()
        input_ = adult.dataset(encoded=False).sample(1)
        with self.assertRaises(TypeError):
            self.instance._realizar_amostragem(input_)
        with self.assertRaises(TypeError):
            self.instance._realizar_amostragem(input_.to_numpy())

    def test_classificar_amostragem(self):
        """`_classificar_amostragem` must return a `np.ndarray` with same share as input"""
        adult = Data.create_new_instance_adult()
        input_ = adult.dataset(encoded=False).sample(1).iloc[0]
        sample = self.instance.sampler.realizar_amostragem(input_, KAOGExpTest.QTD_AMOSTRAS)
        y = self.instance._classificar_amostragem(sample)
        self.assertEqual(sample.shape[0], sample.shape[0])

    def test_criar_kaog(self):
        adult = Data.create_new_instance_adult()
        input_ = adult.dataset(encoded=False).sample(1).iloc[0]
        sample = self.instance.sampler.realizar_amostragem(input_, KAOGExpTest.QTD_AMOSTRAS)
        y = self.instance._classificar_amostragem(sample)
        df = pd.concat([sample, pd.Series(y, name=NOME_COLUNA_Y)], axis=1)
        kaog = self.instance._criar_kaog(df)
        self.assertIsInstance(kaog, KAOG)

    def test_explicar_counterfactual(self):
        adult = Data.create_new_instance_adult()
        input_ = adult.dataset(encoded=False).iloc[0]
        metodo = Counterfactual
        explicacao = self.instance.explicar(input_, metodo, classe_desejada=1)
        self.assertIsInstance(explicacao, metodo)

    def test_validade(self):
        """Verifica se a porcentagem de valores encontrados está dentro do esperado"""
        iris = Data.iris_dataset()
        iris.drop_duplicates(inplace=True)
        validacao = iris.sample(frac=0.3)
        classe_buscada = 2
        train = iris.drop(validacao.index)
        kaog = KAOG(train)

        results = []
        for idx, item in validacao.iterrows():
            instance = Counterfactual(kaog, item, classe_buscada)
            try:
                _ = instance.instancia_modificada
                results.append(True)
            except AttributeError:
                results.append(False)

        # Representa a porcentagem de acertos esperado
        taxa_validacao = 0.9
        porcentagem = sum(results) / len(results)
        self.assertGreaterEqual(porcentagem, taxa_validacao)

if __name__ == '__main__':
    unittest.main()
