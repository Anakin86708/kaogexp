import unittest
from unittest import expectedFailure

import pandas as pd
from kaog import KAOG

from data.loader import ColunaYSingleton
from data.loader.DatasetFromMemory import DatasetFromMemory
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
        pd.testing.assert_index_equal(input_.index.drop(ColunaYSingleton().NOME_COLUNA_Y), sample.columns)

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
        df = pd.concat([sample, pd.Series(y, name=ColunaYSingleton().NOME_COLUNA_Y)], axis=1)
        kaog = self.instance._criar_kaog(df)
        self.assertIsInstance(kaog, KAOG)

    def test_explicar_counterfactual(self):
        adult = Data.create_new_instance_adult()
        input_ = adult.dataset(encoded=False).iloc[0]
        metodo = Counterfactual
        explicacao = self.instance.explicar(input_, metodo, classe_desejada=1)
        self.assertIsInstance(explicacao, metodo)

    def test_explicar_dados_diferentes_iris(self):
        """
        Dada uma instância nunca vista (com dados categóricos já conhecidos), deve ser dada uma explicação válida.
        """
        classe_atual = 0
        classe_desejada = 1
        iris = Data.create_new_instance_iris()
        data = iris.dataset(encoded=False)
        input_ = data[data[ColunaYSingleton().NOME_COLUNA_Y] == classe_atual].sample(1).iloc[0]
        metodo = Counterfactual
        input_index = input_.name
        train_data = data.drop(input_index)

        modelo = RandomForestModel(iris.x().drop(input_index), iris.y().drop(input_index), iris.tratador)
        sampler = LatinSampler(KAOGExpTest.EPSILON)
        instance = KAOGExp(DatasetFromMemory(train_data, pd.Index([])), modelo, sampler)

        result = instance.explicar(input_, metodo, classe_desejada=classe_desejada,
                                   normalizador_associado=iris.normalizador,
                                   tratador_associado=iris.tratador)
        pd.testing.assert_series_equal(input_, result.instancia_original)
        self.assertIsInstance(result, metodo)
        self.assertEqual(classe_desejada, result.classe_desejada)
        self.assertEqual(classe_desejada,
                         modelo.predict(result.instancia_modificada.drop(ColunaYSingleton().NOME_COLUNA_Y)))

    def test_explicat_dados_diferentes_adult(self):
        """
        Dada uma instância nunca vista (com dados categóricos já conhecidos), deve ser dada uma explicação válida.
        """
        classe_atual = 0
        classe_desejada = 1
        adult = Data.create_new_instance_adult()
        data = adult.dataset(encoded=False)
        input_ = data[data[ColunaYSingleton().NOME_COLUNA_Y] == classe_atual].sample(1).iloc[0]
        metodo = Counterfactual
        input_index = input_.name
        train_data = data.drop(input_index)

        modelo = RandomForestModel(adult.x(encoded=True).drop(input_index), adult.y().drop(input_index), adult.tratador)
        sampler = LatinSampler(KAOGExpTest.EPSILON)
        instance = KAOGExp(DatasetFromMemory(train_data, adult._nomes_colunas_categoricas), modelo, sampler)

        result = instance.explicar(input_, metodo, classe_desejada=classe_desejada,
                                   normalizador_associado=adult.normalizador,
                                   tratador_associado=adult.tratador)
        pd.testing.assert_series_equal(input_, result.instancia_original)
        self.assertIsInstance(result, metodo)
        self.assertEqual(classe_desejada, result.classe_desejada)
        modificada = adult.tratador.encode(result.instancia_modificada.drop(ColunaYSingleton().NOME_COLUNA_Y))
        self.assertEqual(classe_desejada, modelo.predict(modificada))

    def test_fixed_cols(self):
        adult = Data.create_new_instance_adult()
        data = adult.dataset(encoded=False)
        input_ = data[data[ColunaYSingleton().NOME_COLUNA_Y] == 0].sample(1).iloc[0]
        metodo = Counterfactual
        input_index = input_.name
        train_data = data.drop(input_index)
        fixed_cols = pd.Index(['age', 'sex', 'hours-per-week'])
        modelo = RandomForestModel(adult.x(encoded=True).drop(input_index), adult.y().drop(input_index), adult.tratador)
        sampler = LatinSampler(KAOGExpTest.EPSILON)
        instance = KAOGExp(DatasetFromMemory(train_data, adult._nomes_colunas_categoricas), modelo, sampler,
                           fixed_cols=fixed_cols)

        result = instance.explicar(input_, metodo, classe_desejada=1,
                                   normalizador_associado=adult.normalizador,
                                   tratador_associado=adult.tratador)

        print("original:\n", result.instancia_original)
        print("\nmodificada:\n", result.instancia_modificada)

        pd.testing.assert_index_equal(fixed_cols, instance.fixed_cols)
        pd.testing.assert_series_equal(input_, result.instancia_original)
        pd.testing.assert_series_equal(input_[fixed_cols], result.instancia_modificada[fixed_cols], check_names=False)

if __name__ == '__main__':
    unittest.main()
