from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from data.treatment.TreatmentAbstract import TreatmentAbstract


class ModelAbstract(ABC):

    def __init__(self, model, tratador: TreatmentAbstract):
        """
        Recebe as informações para realizaremos a predição.

        :param model: Modelo para realizar a predição.
        """
        self._raw_model = model
        self.tratador = tratador

    @property
    def raw_model(self):
        return self._raw_model

    def encode(self, x: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        return self.tratador.encode(x)

    @abstractmethod
    def predict(self, x: Union[pd.Series, pd.DataFrame]) -> Union[int, np.ndarray]:
        """
        Realiza a predição utilizando o modelo associado a classe. O retorno será de acordo com o tipo de dados em `x`,
        sendo que o retorno será um inteiro para um único valor ou um array de inteiros para um conjunto de valores.

        :param x: Dados para realizar a predição.
        :type x: Union[pd.Series, pd.DataFrame]
        :return: Classificação da predição.
        :rtype: Union[int, np.ndarray]
        """
        raise NotImplementedError
