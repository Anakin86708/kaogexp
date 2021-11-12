from random import randint
from typing import Union

import lhsmdu as lhsmdu
import numpy as np
import pandas as pd

from kaogexp.data.loader import NOME_COLUNA_Y
from kaogexp.data.sampler.SamplerAbstract import SamplerAbstract


class LatinSampler(SamplerAbstract):
    """
    Latin Sampler is a sampler that samples from a Latin hypercube using `lhsmdu` method.
    """

    def __init__(self, epsilon: Union[float, np.ndarray], seed: int = None):
        """
        Initializes the Latin Sampler.

        :param epsilon: Represents the limit of the sampling in each feature. Use a float to represent a limit in all
         features. Use a numpy array to represent a limit in each feature.
        :type epsilon: Union[float, np.ndarray]
        :param seed: Seed to be used in the `lhsmdu`.
        :type seed: int
        """
        self.epsilon = epsilon
        if seed is None:
            seed = randint(0, (2 ** 32) - 1)
        self.seed = seed
        lhsmdu.setRandomSeed(seed)

    def realizar_amostragem(self, interest_point: pd.Series, num_samples: int) -> pd.DataFrame:
        """
        Realizes a Latin Hypercube Sampling around 'interest_point' with 'num_samples' samples.
        **Only numerical columns are considered.**

        :param interest_point: Point of interest to be sampled around.
        :type interest_point: pd.Series
        :param num_samples: Number of samples to be generated.
        :type num_samples: int
        :return: DataFrame with samples. Each row is a sample.
        :rtype: pd.DataFrame
        """
        # Prepara os dados
        interest_point = interest_point.copy().drop(NOME_COLUNA_Y, errors='ignore')
        values_to_change = self._get_values_can_change(interest_point)
        interest_point_np = self._sanitize(interest_point).to_numpy()

        # Calcula a amostra e os valores para realizar a transformação
        latin_sample = np.array(lhsmdu.sample(interest_point_np.shape[0], num_samples, randomSeed=self.seed))
        min_ = (interest_point_np - self.epsilon) * values_to_change
        max_ = (interest_point_np + self.epsilon) * values_to_change

        # Realiza a transformação para colocar `latin_sample` ao redor de `interest_point_np`
        sample = map(lambda x: min_ + x * (max_ - min_), latin_sample.T)

        # Reinserir os valores que não puderam ser alterados
        data_frame = self._reinsert_categorical_data(interest_point, sample)
        return data_frame

    def _reinsert_categorical_data(self, interest_point, sample):
        data_frame = pd.DataFrame(sample, columns=interest_point.index).convert_dtypes()
        non_numeric_cols = self._get_non_numeric_indexes(interest_point)
        data_frame[non_numeric_cols] = interest_point[non_numeric_cols]
        data_frame[non_numeric_cols] = data_frame[non_numeric_cols].astype("category")
        return data_frame

    def _get_values_can_change(self, interest_point: pd.Series) -> np.ndarray:
        """
        Gets the values that can change in the input, being the ones that are not numeric.

        :param interest_point: Point of interest.
        :return: Array with the values that can change being True.
        """
        non_numeric_cols = self._get_non_numeric_indexes(interest_point)
        values_to_change = np.array([x not in non_numeric_cols for x in interest_point.index])
        return values_to_change

    @staticmethod
    def _sanitize(input_: pd.Series) -> pd.Series:
        """
        Sanitizes the input to be a pd.Series, replacing string with zeros.

        :param input_: Series to be sanitized.
        :type input_: pd.Series
        :return: Series sanitized.
        :rtype: pd.Series
        """
        input_ = input_.copy()
        non_numeric_cols = LatinSampler._get_non_numeric_indexes(input_)
        input_[non_numeric_cols] = 0
        return input_.convert_dtypes()

    @staticmethod
    def _get_non_numeric_indexes(input_: pd.Series) -> pd.Index:
        """Indexes for columns that are not numeric"""
        try:
            return input_.str.replace(r'[0-9]+', '', regex=True).dropna().index
        except AttributeError:
            return pd.Index([])
