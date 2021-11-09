from abc import ABC, abstractmethod

import pandas as pd


class NormalizerAbstract(ABC):

    def __init__(self, nomes_colunas_numericas: pd.Index):
        self._nomes_colunas_numericas = nomes_colunas_numericas

    @property
    def nomes_colunas_numericas(self):
        return self._nomes_colunas_numericas

    @abstractmethod
    def transform(self, instancia: pd.Series) -> pd.Series:
        pass

    @abstractmethod
    def inverse_transform(self, instancia: pd.Series) -> pd.Series:
        pass
