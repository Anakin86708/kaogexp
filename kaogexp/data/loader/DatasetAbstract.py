from abc import ABC
from typing import Tuple

import pandas as pd

from kaogexp.data.normalizer.NormalizerAbstract import NormalizerAbstract
from kaogexp.data.normalizer.NormalizerFactory import NormalizerFactory
from kaogexp.data.treatment.TreatmentAbstract import TreatmentAbstract
from kaogexp.data.treatment.TreatmentFactory import TreatmentFactory


class DatasetAbstract(ABC):
    NOME_COLUNA_Y = 'target'

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
        if tratar_na:
            self._tratador: TreatmentAbstract = TreatmentFactory.create(tratador)
            self._dataset = TreatmentAbstract.tratar_na(data)
        else:
            self._tratador = None
            if self._dataset.isna().any().any():
                raise Exception("Não é possível continuar com valores faltantes sem tratamento")

        # Normalizar dados se necessário
        if normalizar:
            self._normalizador: NormalizerAbstract = NormalizerFactory.create(normalizador)
        else:
            self._normalizador = None

    def _atribuir_colunas_categoricas(self):
        """Define as colunas necessárias com o tipo adequado de categórico do Pandas"""
        colunas_categoricas = self._dataset.columns[self._nomes_colunas_categoricas]
        self._dataset[colunas_categoricas] = self._dataset[colunas_categoricas].astype("category")
