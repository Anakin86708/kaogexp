from random import randint
from typing import Union

import lhsmdu as lhsmdu
import numpy as np

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

    def realizar_amostragem(self, interest_point: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Realizes a Latin Hypercube Sampling around 'interest_point' with 'num_samples' samples.

        :param interest_point: Point of interest to be sampled around.
        :type interest_point: np.ndarray
        :param num_samples: Number of samples to be generated.
        :type num_samples: int
        :return: Array of samples. Each item is a sample. Use `sample.T[i]` to get all results from all samples from one
         dimension.
        :rtype: np.ndarray
        """
        interest_point = self._sanitize(interest_point)

        # Calcula a amostra e os valores para realizar a transformação
        latin_sample = np.array(lhsmdu.sample(interest_point.shape[0], num_samples, randomSeed=self.seed))
        min_ = interest_point - self.epsilon
        max_ = interest_point + self.epsilon

        # Realiza a transformação para colocar `latin_sample` ao redor de `interest_point`
        sample = np.array(list(map(lambda x: min_ + x * (max_ - min_), latin_sample.T)))

        return sample

    @staticmethod
    def _sanitize(input_: np.ndarray) -> np.ndarray:
        """
        Sanitizes the point of interest to be a numpy array without strings and with dtype float.

        :param input_: Array to be sanitized
        :type input_: np.ndarray
        :return: Array without strings and with dtype float
        :rtype: np.ndarray
        """

        def filter_non_float(x):
            try:
                return float(x)
            except ValueError:
                return float(0)

        filter_non_float = np.vectorize(filter_non_float)

        return filter_non_float(input_)
