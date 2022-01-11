import numpy as np
import pandas as pd
from distython.HEOM import HEOM


class MyHEOM(HEOM):
    """ Extensão do HEOM que permite seu funcionamento com o KAOG e dados categóricos em string"""

    def __init__(self, X: pd.DataFrame, cat_cols, nan_equivalents=(np.nan, 0), normalised="normal"):
        self.cat_cols = cat_cols

        X = X.copy()
        X = self._converter_categoricos(X)
        cat_ix = self._nomes_colunas_para_index(X, cat_cols)
        super().__init__(X, cat_ix, nan_equivalents, normalised)

    def _nomes_colunas_para_index(self, X, cat_cols):
        return [X.columns.get_loc(col) for col in cat_cols]

    def _converter_categoricos(self, X: pd.DataFrame):
        """ Dados em colunas categóricas serão colocados como valores numéricos, já que o HEOM não consegue calcular `range` com valores
        em str.
        """
        for col in self.cat_cols:
            X[col] = pd.factorize(X[col])[0] + 1

        return X

    def heom(self, x, y):
        df = pd.DataFrame([x, y])
        df = self._converter_categoricos(df)
        x = df.iloc[0].to_numpy()
        y = df.iloc[1].to_numpy()

        if len(self.cat_ix) == 0:
            return self._calc_without_cat(x, y)
        return super().heom(x, y)

    def _calc_without_cat(self, x, y):
        # code from HEOM
        # Get indices for missing values, if any
        nan_x_ix = np.flatnonzero(np.logical_or(np.isin(x, self.nan_eqvs), np.isnan(x)))
        nan_y_ix = np.flatnonzero(np.logical_or(np.isin(y, self.nan_eqvs), np.isnan(y)))
        nan_ix = np.unique(np.concatenate((nan_x_ix, nan_y_ix)))
        # Get numerical indices without missing values elements
        num_ix = np.setdiff1d(self.col_ix, self.cat_ix)
        num_ix = np.setdiff1d(num_ix, nan_ix)
        # Calculate the distance for numerical elements
        return np.sum(np.square(np.abs(x[num_ix] - y[num_ix]) / self.range[num_ix]))
