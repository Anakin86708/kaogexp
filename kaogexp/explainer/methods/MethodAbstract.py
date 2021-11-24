from abc import ABC, abstractmethod

import pandas as pd
from kaog import KAOG


class MethodAbstract(ABC):

    def __init__(self, kaog: KAOG, instancia_explicada: pd.Series):
        self.kaog = kaog
        self._instancia_original = instancia_explicada

    @property
    def instancia_original(self) -> pd.Series:
        return self._instancia_original.copy()

    @property
    @abstractmethod
    def instancia_modificada(self) -> pd.Series:
        raise NotImplementedError
