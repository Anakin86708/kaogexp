import unittest
from unittest import expectedFailure

import pandas as pd

from kaogexp.data.sampler.LatinSampler import LatinSampler
from kaogexp.explainer.KAOGExp import KAOGExp
from kaogexp.model.RandomForestModel import RandomForestModel
from util import Data


class KAOGExpTest(unittest.TestCase):
    EPSILON = 0.05

    def setUp(self) -> None:
        self.adult = Data.create_new_instance_adult()
        model = RandomForestModel.from_dataset(self.adult)
        sampler = LatinSampler(KAOGExpTest.EPSILON)
        self.instance = KAOGExp(self.adult, model, sampler)

    def test_instance_compatibility_dataframe(self):
        """Instance must be `pd.Series` or `pd.DataFrame` and with same columns as the `tratador`"""
        adult = Data.create_new_instance_adult()
        input_ = adult.dataset(encoded=False).sample(5)
        self.instance._assert_instance_compatibility(input_)

    def test_instance_compatibility_series(self):
        """Instance must be `pd.Series` or `pd.DataFrame` and with same columns as the `tratador`"""
        adult = Data.create_new_instance_adult()
        input_ = adult.dataset(encoded=False).sample(5).iloc[0]
        self.instance._assert_instance_compatibility(input_)

    @expectedFailure
    def test_instance_compatibility_error(self):
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
        pd.testing.assert_index_equal(input_.index, sample.columns)

    def test_realizar_amostragem_dataframe(self):
        """Realizar amostragem must recive `pd.Series` only."""
        adult = Data.create_new_instance_adult()
        input_ = adult.dataset(encoded=False).sample(1)
        with self.assertRaises(TypeError):
            self.instance._realizar_amostragem(input_)
        with self.assertRaises(TypeError):
            self.instance._realizar_amostragem(input_.to_numpy())


if __name__ == '__main__':
    unittest.main()
