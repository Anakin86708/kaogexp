from abc import ABC
from functools import cached_property
from typing import Tuple

import pandas as pd

from kaogexp.data.loader import NOME_COLUNA_Y
from kaogexp.data.normalizer.NormalizerAbstract import NormalizerAbstract
from kaogexp.data.normalizer.NormalizerFactory import NormalizerFactory
from kaogexp.data.treatment.TreatmentAbstract import TreatmentAbstract
from kaogexp.data.treatment.TreatmentFactory import TreatmentFactory


class DatasetAbstract(ABC):

    def __init__(self,
                 data: pd.DataFrame,
                 colunas_categoricas: pd.Index,
                 valores_na: Tuple[str, ...] = ('?',),
                 tratar_na: bool = True,
                 normalizador: str = "MinMaxNormalizer",
                 tratador: str = "DatasetTreatment"
                 ):

        self._dataset = data
        self._nomes_colunas_categoricas = colunas_categoricas
        self._atribuir_colunas_categoricas()
        self._valores_na = valores_na

        # Tratar valores faltantes se necessário
        self._tratador = None
        if tratar_na:
            self._tratador: TreatmentAbstract = TreatmentFactory.create(tratador, dataset=self._dataset)
            self._dataset = self._tratador.tratar_na(data, self._valores_na)

        if self._dataset.isna().any().any():
            raise Exception("Não é possível continuar com valores faltantes sem tratamento")

        x = self.x(False, False)
        nomes_cols_num = x.drop(self._nomes_colunas_categoricas, axis=1).columns
        self._normalizador: NormalizerAbstract = NormalizerFactory.create(normalizador,
                                                                          x=x,
                                                                          nomes_colunas_normalizar=nomes_cols_num)

    def dataset(self, normalizado: bool = True, encoded: bool = False) -> pd.DataFrame:
        """
        Retorna o dataset normalizado e/ou codificado

        :param normalizado: Se o dataset deve ser normalizado
        :type normalizado: bool
        :param encoded: Se o dataset deve ser codificado
        :type encoded: bool
        :return: O dataset normalizado e/ou codificado
        :rtype: pd.DataFrame
        """
        dataset = self._dataset.copy()
        if normalizado:
            ds_normalizado = self._normalizador.transform(dataset.drop(NOME_COLUNA_Y, axis=1))
            dataset = ds_normalizado.join(dataset[NOME_COLUNA_Y])
        if encoded:
            # TODO: implementar encoding
            pass
        return dataset

    @property
    def tratador(self):
        return self._tratador

    @property
    def normalizador(self):
        return self._normalizador

    @property
    def nomes_colunas_categoricas(self):
        return self._nomes_colunas_categoricas.copy()

    @cached_property
    def nomes_colunas_numericas(self):
        return self.x(False).drop(self._nomes_colunas_categoricas, axis=1).copy().columns

    def x(self, normalizado: bool = True, encoded: bool = True) -> pd.DataFrame:
        return self.dataset(normalizado, encoded).drop(NOME_COLUNA_Y, axis=1)

    def y(self) -> pd.Series:
        return self.dataset(False)[NOME_COLUNA_Y].copy()

    def _atribuir_colunas_categoricas(self) -> None:
        """Define as colunas necessárias com o tipo adequado de categórico do Pandas"""
        colunas_categoricas = self._nomes_colunas_categoricas
        self._dataset[colunas_categoricas] = self._dataset[colunas_categoricas].astype("category")
