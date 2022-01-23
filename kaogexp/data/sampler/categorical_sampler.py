import pandas as pd


class RandomCategoricalSampler:

    def __init__(self, data: pd.DataFrame, cat_cols: pd.Index, fixed_cols: pd.Index = pd.Index([])):
        """
        Utiliza os dados para adquirir as possibilidades de valores categóricos a serem atribuídos em cada coluna.

        :param data: Conjunto de dados, que pode até mesmo incluir valores numéricos
        (dos quais não serão utilizados aqui)
        :type data: pd.DataFrame
        :param cat_cols: Colunas categóricas que podem ser alteradas pela amostragem.
        :type cat_cols: pd.Index
        :param fixed_cols: Colunas que não devem ser alteradas. Pode incluir colunas numéricas, mas serào filtradas
        para uso dessa classe.
        """
        self.data = data.copy()
        self.cat_cols = cat_cols.copy()
        self.fixed_cols = fixed_cols.copy()

        self.colunas_alteradas = self._definir_colunas_alteradas()

    def realizar_amostragem(self, amostragem: pd.DataFrame) -> pd.DataFrame:
        pass

    def _definir_colunas_alteradas(self):
        """
        Com base nas colunas categóricas e colunas fixas, determina apenas aquelas que podem ser alteradas.
        :return: Colunas categóricas que podem ser alteradas
        :rtype: pd.Index
        """
        cat_cols = self.cat_cols.tolist()
        fix_cols = self.fixed_cols.tolist()
        return pd.Index(list(filter(lambda x: x not in fix_cols, cat_cols)))
