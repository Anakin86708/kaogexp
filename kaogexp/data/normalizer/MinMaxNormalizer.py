import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from kaogexp.data.normalizer.NormalizerAbstract import NormalizerAbstract


class MinMaxNormalizer(NormalizerAbstract):
    normalizador = MinMaxScaler()

    def __init__(self, x: pd.DataFrame, nomes_colunas_numericas: pd.Index):
        super().__init__(nomes_colunas_numericas)
        self._data = x
        self.normalizador.fit(x.loc[:, self.nomes_colunas_numericas])

    def normalizar(self, instancia: pd.Series) -> pd.Series:
        """
        Normaliza uma instância utilizando o normalizador, já preparado nesta classe.

        :param instancia: Instância a ser normalizada, representando uma linha do conjunto de dados.
        :type instancia: pd.Series
        :return: Instância normalizada.
        :rtype: pd.Series
        """
        # TODO: garantir que o formato de saída ainda seja a Series correta
        return self.normalizador.transform(instancia.loc[self.nomes_colunas_numericas].values.reshape(1, -1))[0]

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
