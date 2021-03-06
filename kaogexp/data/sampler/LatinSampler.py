import warnings
from random import randint
from typing import Union

import lhsmdu as lhsmdu
import numpy as np
import pandas as pd

from kaogexp.data.loader import ColunaYSingleton
from kaogexp.data.sampler.SamplerAbstract import SamplerAbstract


class LatinSampler(SamplerAbstract):
    """
    Latin Sampler is a sampler that samples from a Latin hypercube using `lhsmdu` method.
    """

    def __init__(self, epsilon: Union[float, np.ndarray], incremento: float = 0.05, limite_epsilon: float = 1.0,
                 seed: int = None):
        """
        Initializes the Latin Sampler.

        :param epsilon: Represents the limit of the sampling in each feature. Use a float to represent a limit in all
         features. Use a numpy array to represent a limit in each feature.
        :type epsilon: Union[float, np.ndarray]
        :param incremento: Represents the increment of the sampling in each feature.
        :type incremento: float
        :param seed: Seed to be used in the `lhsmdu`.
        :type seed: int
        """
        self.incremento = incremento
        self._initial_epsilon = epsilon
        self._limite_epsilon = limite_epsilon
        self._epsilon = epsilon
        if seed is None:
            seed = randint(0, (2 ** 32) - 1)
        self.seed = seed
        lhsmdu.setRandomSeed(seed)

    @property
    def epsilon(self) -> float:
        return round(self._epsilon, 6)

    @property
    def fixed_cols(self) -> pd.Index:
        if hasattr(self, '_fixed_cols') and self._fixed_cols is not None:
            return self._fixed_cols.copy()
        return pd.Index([])

    @fixed_cols.setter
    def fixed_cols(self, fixed_cols: pd.Index):
        self._fixed_cols = fixed_cols.copy() if fixed_cols is not None else None

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
        interest_point = interest_point.copy().drop(ColunaYSingleton().NOME_COLUNA_Y, errors='ignore')
        interest_point_np = self._sanitize(interest_point).to_numpy().astype(float)

        # Calcula a amostra e os valores para realizar a transforma????o
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
            latin_sample = np.array(lhsmdu.sample(interest_point_np.shape[0], num_samples, randomSeed=self.seed))
        data_frame = self._prepare_sample(interest_point, interest_point_np, latin_sample, num_samples)
        return data_frame

    def increase_epsilon(self) -> None:
        """Used to increase the epsilon value."""
        if (self.epsilon + self.incremento) > self._limite_epsilon:
            self.reset_epsilon()
            raise ValueError("Epsilon value cannot be greater than {}".format(self._limite_epsilon))
        if isinstance(self.epsilon, float):
            self._epsilon += self.incremento
        else:
            mask = self.epsilon.copy()
            mask[mask != 0] = self.incremento
            self._epsilon += mask

    def reset_epsilon(self) -> None:
        """Retorna o epsilon para o valor inicial."""
        self._epsilon = self._initial_epsilon

    def _prepare_sample(self, interest_point, interest_point_np, latin_sample, num_samples):
        """
        Com base nos dados de `interest_point` e `interest_point_np`, coloca o amostra no intervalo desejado, al??m de
        colocar os valores de index, de forma que o primeiro valor seja o do ponto de interesse.

        :param interest_point: Ponto de interesse.
        :type interest_point: pd.Series
        :param interest_point_np: Ponto de interesse em formato numpy.
        :type interest_point_np: np.ndarray
        :param latin_sample: Amostra gerada pelo lhsmdu.
        :type latin_sample: np.ndarray
        :param num_samples: N??mero de amostras que foram geradas.
        :type num_samples: int
        :return: DataFrame com as amostras.
        :rtype: pd.DataFrame
        """
        values_to_change = self._get_values_can_change(interest_point)
        min_ = (interest_point_np - self.epsilon) * values_to_change
        max_ = (interest_point_np + self.epsilon) * values_to_change
        min_, max_ = self._restrain_limits(min_, max_)
        # Realiza a transforma????o para colocar `latin_sample` ao redor de `interest_point_np`
        sample = map(lambda x: min_ + x * (max_ - min_), latin_sample.T)
        # Reinserir os valores que n??o puderam ser alterados
        data_frame = self._reinsert_categorical_data(interest_point, sample)
        data_frame = self._reinsert_fixed_cols(interest_point, data_frame)
        self._reindex_data(data_frame, interest_point, num_samples)
        return data_frame

    @staticmethod
    def _restrain_limits(min_: np.ndarray, max_: np.ndarray):
        """
        Keep limits inside [0, 1].

        :param min_: Inferior limit.
        :type min_: np.ndarray
        :param max_: Superior limit.
        :type max_: np.ndarray
        :return:
        """
        min_ = np.array(list(map(lambda x: 0. if x < 0 else x, min_)))
        max_ = np.array(list(map(lambda x: 1. if x > 1 else x, max_)))
        return min_, max_

    def _reindex_data(self, data_frame, interest_point, num_samples):
        """Colocar valores de index a partir do index do ponto de interesse"""
        try:
            start = int(interest_point.name) + 1
        except TypeError:
            start = 1
        end = start + num_samples
        data_frame.index = range(start, end)

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
        values_to_change = np.array(
            [x not in non_numeric_cols and x not in self.fixed_cols for x in interest_point.index])
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

    def _reinsert_fixed_cols(self, interest_point: pd.Series, data_frame: pd.DataFrame):
        data_frame[self.fixed_cols] = interest_point[self.fixed_cols]
        return data_frame
