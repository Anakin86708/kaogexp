from abc import ABC
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
                 valores_na: Tuple[str, ...] = ('?'),
                 tratar_na: bool = True,
                 normalizador: str = "MinMaxNormalizer",
                 normalizar: bool = True,
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

        # Normalizar dados se necessário
        self._normalizador = None
        if normalizar:
            x = self.x(False, False)
            nomes_cols_num = x.drop(self._nomes_colunas_categoricas, axis=1).columns
            self._normalizador: NormalizerAbstract = NormalizerFactory.create(normalizador,
                                                                              x=x,
                                                                              nomes_colunas_numericas=nomes_cols_num)
            # TODO: aplicar a normalizacao no dataset

    @property
    def dataset(self):
        return self._dataset.copy()

    @property
    def tratador(self):
        return self._tratador

    @property
    def normalizador(self):
        return self._normalizador

    @property
    def nomes_colunas_categoricas(self):
        return self._nomes_colunas_categoricas.copy()

    def x(self, normalizado: bool = True, encoded: bool = True) -> pd.DataFrame:
        x: pd.DataFrame = self._dataset.drop(NOME_COLUNA_Y, axis=1)
        if normalizado:
            x.apply(self._normalizador.transform, inplace=True)
        if encoded:
            x.apply(self._tratador.encode, inplace=True)
        return x

    def _atribuir_colunas_categoricas(self):
        """Define as colunas necessárias com o tipo adequado de categórico do Pandas"""
        colunas_categoricas = self._nomes_colunas_categoricas
        self._dataset[colunas_categoricas] = self._dataset[colunas_categoricas].astype("category")
