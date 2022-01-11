from typing import Tuple

import pandas as pd

from kaogexp.data.loader.DatasetAbstract import DatasetAbstract


class DatasetFromMemory(DatasetAbstract):
    """
    Representação dos dados, mas que pode ser criado diretamente a partir de um DataFrame.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 colunas_categoricas: pd.Index,
                 valores_na: Tuple[str, ...] = ('?',),
                 tratar_na: bool = True,
                 normalizador: str = "MinMaxNormalizer",
                 tratador: str = "DatasetTreatment"
                 ):
        super().__init__(data, colunas_categoricas, valores_na, tratar_na, normalizador, tratador)
