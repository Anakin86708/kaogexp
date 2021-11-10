from typing import Tuple

import numpy as np
import pandas as pd

from kaogexp.data.loader import NOME_COLUNA_Y
from kaogexp.data.treatment.TreatmentAbstract import TreatmentAbstract

SUBSTITUTO_NA = np.nan


class DatasetTreatment(TreatmentAbstract):

    def __init__(self, dataset: pd.DataFrame):
        nomes_colunas_originais = dataset.columns
        nomes_colunas_encoded = pd.get_dummies(dataset.drop(NOME_COLUNA_Y, axis=1, errors='ignore')).columns
        super().__init__(nomes_colunas_originais, nomes_colunas_encoded)

    def encode(self, instancia: pd.Series) -> pd.Series:
        dummies = pd.get_dummies(instancia)
        pass

    def decode(self, instancia: pd.Series) -> pd.Series:
        pass

    @staticmethod
    def tratar_na(dataset: pd.DataFrame, valores_na: Tuple) -> pd.DataFrame:
        """
        Substitui os valores faltantes, definidos como `valores_na`, por `DatasetTreatment.SUBSTITUTO_NA`.
        Em seguida é feita uma interpolação dos dados faltantes.

        :param dataset: Conjunto de dados a ser tratado.
        :type dataset: pd.DataFrame
        :param valores_na: Valores faltantes a serem substituídos.
        :type valores_na: Tuple
        :return: Conjunto de dados tratado, sem valores faltantes.
        :rtype: pd.DataFrame
        """
        dataset = dataset.replace(valores_na, SUBSTITUTO_NA)
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
        :raise ValueError: Se restar algum valor NA do conjunto de dados.
        """
        categoricas_and_y = DatasetTreatment._obter_cols_categoricas_and_y(dataset)
        numericas = dataset.drop(categoricas_and_y, axis=1).columns

        dataset[categoricas_and_y] = dataset[categoricas_and_y].fillna(method='ffill')
        dataset[categoricas_and_y] = dataset[categoricas_and_y].fillna(method='bfill')
        dataset[numericas] = dataset[numericas].astype('float')
        dataset[numericas] = dataset[numericas].interpolate(method='linear', limit_direction='both')

        if dataset.isna().any().any():
            index_nans = pd.isnull(dataset).any(1).to_numpy().nonzero()[0].tolist()
            raise ValueError(
                f'Falha ao executar a interpolação: Existem valores faltantes na instância.\nIndex:{index_nans}')

        return dataset

    @staticmethod
    def _obter_cols_categoricas_and_y(dataset):
        return dataset.select_dtypes(['category']).columns
