from typing import Tuple

import pandas as pd

from kaogexp.data.loader.DatasetAbstract import DatasetAbstract


class DatasetFromFile(DatasetAbstract):

    def __init__(
            self,
            path_file: str,
            colunas_categoricas: pd.Index,
            delimiter: str = ',',
            valores_na: Tuple[str, ...] = ('?',),
            tratar_na: bool = True,
            normalizador: str = "MinMaxNormalizer",
            tratador: str = "DatasetTreatment",
            header: int = 0,
    ):
        with open(path_file, 'r') as file:
            data = pd.read_csv(
                file,
                delimiter=delimiter,
                na_values=valores_na,
                header=header,
            )
        super().__init__(data, colunas_categoricas, valores_na, tratar_na, normalizador, tratador)
