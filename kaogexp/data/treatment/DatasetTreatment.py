from typing import Tuple

import numpy as np
import pandas as pd

from kaogexp.data.treatment.TreatmentAbstract import TreatmentAbstract


class DatasetTreatment(TreatmentAbstract):
    SUBSTITUTO_NA = np.nan

    def __init__(self, nomes_colunas_originais: pd.Index, nomes_colunas_encoded: pd.Index):
        super().__init__(nomes_colunas_originais, nomes_colunas_encoded)

    def encode(self, instancia: pd.Series) -> pd.Series:
        pass

    def decode(self, instancia: pd.Series) -> pd.Series:
        pass

    @staticmethod
    def tratar_na(dataset: pd.DataFrame, valores_na: Tuple[str, ...]) -> pd.DataFrame:
        """
        Substitui os valores faltantes, definidos como `valores_na`, por `DatasetTreatment.SUBSTITUTO_NA`.
        Em seguida é feita uma interpolação dos dados faltantes.

        :param dataset: Conjunto de dados a ser tratado.
        :type dataset: pd.DataFrame
        :param valores_na: Valores faltantes a serem substituídos.
        :type valores_na: Tuple[str, ...]
        :return: Conjunto de dados tratado, sem valores faltantes.
        :rtype: pd.DataFrame
        """
        dataset = dataset.replace(valores_na, DatasetTreatment.SUBSTITUTO_NA)
        return DatasetTreatment._interpolar_na(dataset)

    @staticmethod
    def _interpolar_na(dataset: pd.DataFrame):
        """Interpola os valores faltantes de uma instância.
        Para valores categóricos, é utilizado o método `pandas.DataFrame.fillna`, no método `ffill`, enquanto valores
        numéricos são interpolados com o método `pandas.DataFrame.interpolate`.

        :param dataset: Conjunto de dados a ser tratado.
        :type dataset: pd.DataFrame
        :return: Conjunto de dados tratado.
        :rtype: pd.DataFrame
        """
        categoricas = dataset.select_dtypes(['category']).columns
        numericas = list(filter(lambda col: col not in categoricas, dataset.columns))
        for col in dataset.columns:
            # Pular colunas que não contenham valores NaN
            if not dataset[col].isna().any():
                continue

            if col in categoricas:
                dataset[col].fillna(method='ffill', inplace=True)
            else:
                dataset[col] = dataset[col].astype(float)
                dataset[col].interpolate(inplace=True)

        return dataset
