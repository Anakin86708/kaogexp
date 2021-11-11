from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SamplerAbstract(ABC):

    @abstractmethod
    def realizar_amostragem(self, ponto_interesse: pd.Series, qtd_amostras: int) -> np.ndarray:
        raise NotImplementedError
