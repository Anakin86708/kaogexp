from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from kaogexp.data.loader import NOME_COLUNA_Y
from kaogexp.data.treatment.TreatmentAbstract import TreatmentAbstract

SUBSTITUTO_NA = np.nan


class DatasetTreatment(TreatmentAbstract):

    def __init__(self, dataset: pd.DataFrame):
        nomes_colunas_encoded, nomes_colunas_originais = self._obter_colunas(dataset)
        super().__init__(nomes_colunas_originais, nomes_colunas_encoded)

    def encode(self, instancia: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """
        Codifica uma instância ou um conjunto de instâncias. Os dados retornados estão em um formato que pode ser
        utilizado pelo modelo para realizar a classificação dos dados.

        :param instancia: Instância ou conjunto de instâncias a ser codificada.
        :type instancia: Union[pd.Series, pd.DataFrame]
        :return: Dados no formado compatível com o modelo, sem a presença de valores em `str`.
        :rtype: Union[pd.Series, pd.DataFrame]
        """
        if isinstance(instancia, pd.Series):
            return self.encode(pd.DataFrame([instancia])).iloc[0]

        elif isinstance(instancia, pd.DataFrame):
            dummies = pd.get_dummies(instancia.drop(NOME_COLUNA_Y, axis=1, errors='ignore'))
            missing_cols = self._get_missing_cols(dummies)
            dummies = dummies.assign(**{col: 0 for col in missing_cols})
            if NOME_COLUNA_Y in instancia.columns:
                dummies = pd.concat([dummies, instancia[NOME_COLUNA_Y]], axis=1)
            assert NOME_COLUNA_Y in dummies if NOME_COLUNA_Y in instancia else NOME_COLUNA_Y not in dummies
            assert self._get_missing_cols(dummies).empty
            return dummies
        else:
            raise TypeError(f'Tipo inválido de instância: {type(instancia)}')
        pass

    def decode(self, instancia: pd.Series) -> pd.Series:
        raise NotImplementedError

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

    def atualizar_colunas(self, dataset: pd.DataFrame) -> None:
        self._nomes_colunas_encoded, self._nomes_colunas_originais = self._obter_colunas(dataset)

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
        categoricas = DatasetTreatment._obter_cols_categoricas(dataset)
        categoricas_and_y = pd.Index(categoricas.to_list() + [NOME_COLUNA_Y])
        numericas = dataset.drop(categoricas_and_y, axis=1).columns

        dataset[categoricas] = dataset[categoricas].fillna(method='ffill')
        dataset[categoricas] = dataset[categoricas].fillna(method='bfill')
        dataset[numericas] = dataset[numericas].astype('float')
        dataset[numericas] = dataset[numericas].interpolate(method='linear', limit_direction='both')

        if dataset.isna().any().any():
            index_nans = pd.isnull(dataset).any(1).to_numpy().nonzero()[0].tolist()
            raise ValueError(
                f'Falha ao executar a interpolação: Existem valores faltantes na instância.\nIndex:{index_nans}')

        return dataset

    def _obter_colunas(self, dataset: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
        """
        Obtém os nomes das colunas do dataset em seu formato original e após aplicada a codificação.

        :param dataset: Conjunto de dados de onde serão extraidas as colunas.
        :type dataset: pd.DataFrame
        :return: Tupla contendo os nomes das colunas original e após a codificação.
        :rtype: Tuple[pd.Index, pd.Index]
        """
        nomes_colunas_originais = dataset.columns
        # Colunas dummy sem a coluna de y
        dummy_columns = pd.get_dummies(dataset.drop(NOME_COLUNA_Y, axis=1, errors='ignore')).columns
        # Adiciona a coluna de y
        nomes_colunas_encoded = pd.Index(dummy_columns.to_list() + [NOME_COLUNA_Y])
        return nomes_colunas_encoded, nomes_colunas_originais

    def _get_missing_cols(self, dummies: pd.DataFrame) -> pd.Index:
        """
        Retorna as colunas que não estão presentes em `dummies`.

        :param dummies: `pd.DataFrame` com colunas a serem verificadas.
        :type dummies: pd.DataFrame
        :return: Colunas que não estão presentes em `dummies` mas que são previstas no formato encoding.
        :rtype: pd.Index
        """
        return self.nomes_colunas_encoded.drop(NOME_COLUNA_Y).difference(dummies.columns)

    @staticmethod
    def _obter_cols_categoricas(dataset):
        columns = dataset.drop(NOME_COLUNA_Y, axis=1, errors='ignore').select_dtypes(['category']).columns
        assert NOME_COLUNA_Y not in columns
        return columns
