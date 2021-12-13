import numpy as np
import pandas as pd
from distython.HEOM import HEOM


class MyHEOM(HEOM):
    """ Extensão do HEOM que permite seu funcionamento com o KAOG e dados categóricos em string"""

    def __init__(self, X: pd.DataFrame, cat_cols, nan_equivalents=(np.nan, 0), normalised="normal"):
        X = self._converter_categoricos(X, cat_cols)
        cat_ix = self._nomes_colunas_para_index(X, cat_cols)
        super().__init__(X, cat_ix, nan_equivalents, normalised)

    def _nomes_colunas_para_index(self, X, cat_cols):
        return [X.columns.get_loc(col) for col in cat_cols]

    def _converter_categoricos(self, X: pd.DataFrame, cat_cols):
        """ Dados em colunas categóricas serão colocados como valores numéricos, já que o HEOM não consegue calcular `range` com valores
        em str.
        """
        for col in cat_cols:
            X[col] = pd.factorize(X[col])[0] + 1

        return X

    def heom(self, x, y):
        df = pd.DataFrame([x, y])
        df = self._converter_categoricos(df, self.cat_ix)
        x = df.iloc[0].to_numpy()
        y = df.iloc[1].to_numpy()
        return super(HEOM, self).heom(x, y)
