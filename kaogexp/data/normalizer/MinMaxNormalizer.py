from typing import Union

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from kaogexp.data.normalizer.NormalizerAbstract import NormalizerAbstract


class MinMaxNormalizer(NormalizerAbstract):
    normalizer = MinMaxScaler()

    def __init__(self, x: pd.DataFrame, nomes_colunas_normalizar: pd.Index):
        super().__init__(nomes_colunas_normalizar)
        self._data = x
        self.normalizer.fit(x.loc[:, self.nomes_colunas_normalizar])

    def transform(self, instancia: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """
        Normaliza uma instância utilizando o normalizer, já preparado nesta classe.

        :param instancia: Instância a ser normalizada, representando uma linha do conjunto de dados.
        :type instancia: Union[pd.Series, pd.DataFrame]
        :return: Instância normalizada, sendo do mesmo tipo de `instancia`.
        :rtype: Union[pd.Series, pd.DataFrame]
        :raise: ValueError: Se o tipo de `instancia` não for um pd.Series ou pd.DataFrame.
        """
        instancia = instancia.copy()
        if isinstance(instancia, pd.Series):
            return self.transform(pd.DataFrame([instancia])).iloc[0]

        elif isinstance(instancia, pd.DataFrame):
            instancia[self.nomes_colunas_normalizar] = self.normalizer.transform(
                instancia[self.nomes_colunas_normalizar])
        return instancia

    def inverse_transform(self, instancia: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """
        Reverte a normalização de uma instância.

        :param instancia: Instância a ser revertida, podendo representar uma linha do conjunto de dados ou todo um
        conjunto de linhas em um `pd.DataFrame`.
        :type instancia: Union[pd.Series, pd.DataFrame]
        :return: Instância revertida, contendo valores condizentes com os dados originais.
        :rtype: Union[pd.Series, pd.DataFrame]
        """
        instancia = instancia.copy()
        if isinstance(instancia, pd.Series):
            return self.inverse_transform(pd.DataFrame([instancia])).iloc[0]
        elif isinstance(instancia, pd.DataFrame):
            instancia[self.nomes_colunas_normalizar] = self.normalizer.inverse_transform(
                instancia[self.nomes_colunas_normalizar])
        return instancia
