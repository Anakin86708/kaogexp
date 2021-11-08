from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd


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

    @abstractmethod
    def encode(self, instancia: pd.Series) -> pd.Series:
        pass

    @abstractmethod
    def decode(self, instancia: pd.Series) -> pd.Series:
        pass

    @staticmethod
    @abstractmethod
    def tratar_na(dataset: pd.DataFrame, valores_na: Tuple[str, ...]) -> pd.DataFrame:
        pass
