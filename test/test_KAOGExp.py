import unittest
from unittest import expectedFailure

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

    def test_instance_copatibility_series(self):
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


if __name__ == '__main__':
    unittest.main()
