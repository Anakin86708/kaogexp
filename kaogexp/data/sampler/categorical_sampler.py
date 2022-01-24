from random import choice
from typing import Dict, List

import pandas as pd


class RandomCategoricalSampler:

    def __init__(self, data: pd.DataFrame, cat_cols: pd.Index, fixed_cols: pd.Index = pd.Index([])):
        """
        Utiliza os dados para adquirir as possibilidades de valores categóricos a serem atribuídos em cada coluna.

        :param data: Conjunto de dados que inclui todas as possibilidades de valores categóricos,
        que pode até mesmo incluir valores numéricos (dos quais não serão utilizados aqui).
        :type data: pd.DataFrame
        :param cat_cols: Colunas categóricas que podem ser alteradas pela amostragem.
        :type cat_cols: pd.Index
        :param fixed_cols: Colunas que não devem ser alteradas. Pode incluir colunas numéricas, mas serào filtradas
        para uso dessa classe.
        """
        self.data = data.copy()
        self.cat_cols = cat_cols.copy()
        self.fixed_cols = fixed_cols.copy() if fixed_cols is not None else pd.Index([])

        self.colunas_alteradas = self._definir_colunas_alteradas()
        self.unique_cat_cols = self._obter_valores_cat_unicos()

    def realizar_amostragem(self, amostragem: pd.DataFrame) -> pd.DataFrame:
        """
        Para cada linha da amostragem, alterar as colunas categóricas possíveis para outro dado categórico.

        :param amostragem: Dados onde serão alteradas as colunas categóricas
        :type amostragem: pd.DataFrame
        :return: Amostra, com dados categóricos alterados aleatóriamente.
        :rtype: pd.DataFrame
        """
        amostragem = amostragem.copy()
        amostragem = amostragem.apply(self._aleatorizar_series, axis=1)
        return amostragem

    def _definir_colunas_alteradas(self):
        """
        Com base nas colunas categóricas e colunas fixas, determina apenas aquelas que podem ser alteradas.
        :return: Colunas categóricas que podem ser alteradas
        :rtype: pd.Index
        """
        cat_cols = self.cat_cols.tolist()
        fix_cols = self.fixed_cols.tolist()
        return pd.Index(list(filter(lambda x: x not in fix_cols, cat_cols)))

    def _obter_valores_cat_unicos(self) -> Dict[str, List]:
        """
        Obtêm os valores únicos de cada coluna categórica que pode ser alterada.

        :return: Dados únicos da cada coluna categórica.
        :rtype: Dict[str, List]
        """
        result = {}
        for col in self.colunas_alteradas:
            result[col] = self.data[col].unique().tolist()
        return result

    def _aleatorizar_series(self, row: pd.Series):
        for col, item in row[self.colunas_alteradas].iteritems():
            row[col] = choice(list(filter(lambda x: x != item, self.unique_cat_cols[col])))
        return row
