from abc import ABC, abstractmethod

import pandas as pd


class SamplerAbstract(ABC):

    @property
    @abstractmethod
    def epsilon(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def realizar_amostragem(self, ponto_interesse: pd.Series, qtd_amostras: int) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def increase_epsilon(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def reset_epsilon(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def fixed_cols(self) -> pd.Index:
        raise NotImplementedError

    @fixed_cols.setter
    @abstractmethod
    def fixed_cols(self, fixed_cols: pd.Index) -> None:
        raise NotImplementedError
