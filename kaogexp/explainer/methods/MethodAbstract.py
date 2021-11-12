from abc import ABC, abstractmethod

import pandas as pd
from kaog import KAOG


class MethodAbstract(ABC):

    def __init__(self, kaog: KAOG):
        self.kaog = kaog

    @property
    @abstractmethod
    def instancia_original(self) -> pd.Series:
        raise NotImplementedError

    @property
    @abstractmethod
    def instancia_modificada(self) -> pd.Series:
        raise NotImplementedError
