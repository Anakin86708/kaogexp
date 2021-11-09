from abc import ABC, abstractmethod

import numpy as np


class SamplerAbstract(ABC):

    @abstractmethod
    def realizar_amostragem(self, ponto_interesse: np.ndarray, qtd_amostras: int) -> np.ndarray:
        raise NotImplementedError
