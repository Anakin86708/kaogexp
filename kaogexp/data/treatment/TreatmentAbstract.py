from abc import ABC, abstractmethod
from typing import Tuple
from typing import Union

import pandas as pd

from data.loader import ColunaYSingleton


class TreatmentAbstract(ABC):
    def __init__(self, nomes_colunas_originais: pd.Index, nomes_colunas_encoded: pd.Index):
        self._nomes_colunas_originais = nomes_colunas_originais
        self._nomes_colunas_encoded = nomes_colunas_encoded

    @property
    def nomes_colunas_originais(self) -> pd.Index:
        return self._nomes_colunas_originais

    @property
    def nomes_colunas_encoded(self) -> pd.Index:
        return self._nomes_colunas_encoded

    @property
    def nomes_colunas_categoricas_encoded(self):
        return pd.Index({*self.nomes_colunas_encoded} - {*self.nomes_colunas_originais}).drop(
            ColunaYSingleton().NOME_COLUNA_Y,
            errors='ignore')

    @abstractmethod
    def encode(self, instancia: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, instancia: pd.Series) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def tratar_na(dataset: pd.DataFrame, valores_na: Tuple) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def atualizar_colunas(self, dataset: pd.DataFrame) -> None:
        raise NotImplementedError
