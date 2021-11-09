from typing import Union

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from kaogexp.data.normalizer.NormalizerAbstract import NormalizerAbstract


class MinMaxNormalizer(NormalizerAbstract):
    normalizador = MinMaxScaler()

    def __init__(self, x: pd.DataFrame, nomes_colunas_numericas: pd.Index):
        super().__init__(nomes_colunas_numericas)
        self._data = x
        self.normalizador.fit(x.loc[:, self.nomes_colunas_numericas])

    def normalizar(self, instancia: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """
        Normaliza uma instância utilizando o normalizador, já preparado nesta classe.

        :param instancia: Instância a ser normalizada, representando uma linha do conjunto de dados.
        :type instancia: Union[pd.Series, pd.DataFrame]
        :return: Instância normalizada, sendo do mesmo tipo de `instancia`.
        :rtype: Union[pd.Series, pd.DataFrame]
        :raise: ValueError: Se o tipo de `instancia` não for um pd.Series ou pd.DataFrame.
        """
        if isinstance(instancia, pd.Series):
            return self.normalizar(pd.DataFrame([instancia])).iloc[0]

        elif isinstance(instancia, pd.DataFrame):
            instancia[self.nomes_colunas_numericas] = self.normalizador.transform(
                instancia[self.nomes_colunas_numericas])
        return instancia

    def reverter(self, instancia: pd.Series) -> pd.Series:
        """
        Reverte a normalização de uma instância.

        :param instancia: Instância a ser revertida, representando uma linha do conjunto de dados.
        :type instancia: pd.Series
        :return: Instância revertida, contendo valores condizentes com os dados originais.
        :rtype: pd.Series
        """
        # TODO: garantir que o formato de saída ainda seja a Series correta
        return self.normalizador.inverse_transform(instancia.loc[self.nomes_colunas_numericas].values.reshape(1, -1))[0]
