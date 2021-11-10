from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


class NormalizerAbstract(ABC):

    def __init__(self, nomes_colunas_normalizar: pd.Index):
        """
        Armazena os nomes das colunas numéricas que serão normalizadas.

        :param nomes_colunas_normalizar: Nomes para colunas numéricas.
        :type nomes_colunas_normalizar: pd.Index
        :return: None
        """
        self._nomes_colunas_normalizar = nomes_colunas_normalizar

    @property
    def nomes_colunas_normalizar(self) -> pd.Index:
        """
        Retorna os nomes das colunas que serão normalizadas.
        :return: Nomes das colunas que serão normalizadas.
        :rtype: pd.Index
        """
        return self._nomes_colunas_normalizar

    @abstractmethod
    def transform(self, instancia: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """
        Aplica a normalização à uma instância, podendo ser uma `pd.Series` ou um `pd.DataFrame`.

        :param instancia: Dado que será normalizado.
        :type instancia: Union[pd.Series, pd.DataFrame]
        :return: Dado normalizado, no mesmo tipo de entrado.
        :rtype: Union[pd.Series, pd.DataFrame]
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, instancia: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """
        Aplica a inversão da normalização à uma instância, podendo ser uma `pd.Series` ou um `pd.DataFrame`.

        :param instancia: Dado normalizado que será invertido.
        :type instancia: Union[pd.Series, pd.DataFrame]
        :return: Dado invertido com os valores originais, no mesmo tipo de entrado.
        :rtype: Union[pd.Series, pd.DataFrame]
        """
        raise NotImplementedError
