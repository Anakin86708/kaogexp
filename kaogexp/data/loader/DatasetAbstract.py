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
        """
        Inicializa o dataset.
        Será feita a atribuição do tipo adequado de categórico do Pandas para as colunas categóricas.
        Trata os dados faltantes em `data`, definidos em `valores_na`, se `tratar_na` for True. Caso False, o valor de
        `tratador` não será usado.
        Finalmente é criado o normalizado com base no `normalizador`.

        
        :param data: Conjunto de dados.
        :type data: pd.DataFrame
        :param colunas_categoricas: Nomes para as colunas categóricas.
        :type colunas_categoricas: pd.Index
        :param valores_na: Valores que serão considerados como N/A.
        :type valores_na: Tuple[str, ...]
        :param tratar_na: Se deve tratar os valores N/A.
        :type tratar_na: bool
        :param normalizador: Nome do normalizador a ser criado pela factory.
        :type normalizador: str
        :param tratador: Nome do tratador a ser criado pela factory.
        :type tratador: str
        """

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
    def nomes_colunas_numericas(self) -> pd.Index:
        """Definidas como as colunas que são são a `NOME_COLUNA_Y` ou colunas categóricas."""
        return self.x(False).drop(self._nomes_colunas_categoricas, axis=1).copy().columns

    def x(self, normalizado: bool = True, encoded: bool = True) -> pd.DataFrame:
        """
        Retorna o dataset sem `NOME_COLUNA_Y`.

        :param normalizado: Se x deven ser normalizado.
        :type normalizado: bool
        :param encoded: Se x deve ser codificado.
        :type encoded: bool
        :return: O dataset sem `NOME_COLUNA_Y`.
        """
        return self.dataset(normalizado, encoded).drop(NOME_COLUNA_Y, axis=1)

    def y(self) -> pd.Series:
        """Apenas `NOME_COLUNA_Y` do dataset."""
        return self.dataset(False)[NOME_COLUNA_Y].copy()

    def _atribuir_colunas_categoricas(self) -> None:
        """Define as colunas necessárias com o tipo adequado de categórico do Pandas."""
        colunas_categoricas = self._nomes_colunas_categoricas
        self._dataset[colunas_categoricas] = self._dataset[colunas_categoricas].astype("category")
