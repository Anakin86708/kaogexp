from typing import Union

import numpy as np
import pandas as pd


class MetricCategorical:
    """
    Measure the distance between instances, accounting for categorical data.

    An instance is encoded with dummies and those labeled as 1 are converted to sqrt(.5). It allows the algorithm to
    perform euclidean distance and categorical distances, being 1 when different and 0 when equal.
    """

    def __init__(self, data: pd.DataFrame, cat_cols: pd.Index):
        """
        :param data: Utilizado para determinar o dummy dos dados.
        :type data: pd.DataFrame
        :param cat_cols: Colunas de `data` que serão tratadas como categóricas.
        :type cat_cols: pd.Index
        """
        self._data = data.copy()
        self._cat_cols = cat_cols.copy()
        self._dummy_cols = pd.get_dummies(data, columns=cat_cols).columns

    @property
    def data(self):
        return self._data.copy()


    @property
    def cat_cols(self):
        return self._cat_cols.copy()

    def calculate(self, x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate the normalized eucliian distance between two series, considering the categorical data distance as
        1 if diferent or 0 if equal.

        :param x: Series with the first data. **DATA MUST BE NORMALIZED**
        :type x: Union[pd.Series, np.ndarray]
        :param y: Series with the second data. **DATA MUST BE NORMALIZED**
        :type y: Union[pd.Series, np.ndarray]
        :return: The normalized eucliian distance between the two series.
        :rtype: float
        """
        if isinstance(x, np.ndarray):
            x = pd.Series(x, index=self.data.columns)
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=self.data.columns)

        if not self.cat_cols.empty:
            x_, y_ = self._apply_dummy(pd.Series(x)), self._apply_dummy(pd.Series(y))
        else:
            x_, y_ = x, y

        if np.array([x_, y_]).shape[0] != 2:
            raise RuntimeError()

        return np.linalg.norm(x_ - y_)

    def _apply_dummy(self, x: pd.Series) -> pd.Series:
        """Apply the dummy function to the series.
        All the categorical data are replaced by sqrt(0.5), and the rest by 0. That allows to calculate the square and
        sum the values to obtain 1 in case of diferent values.

        :param x: Series to be applied.
        :type x: pd.Series
        :return: Series with one-hot encoding.
        :rtype: pd.Series
        """
        dummies = pd.get_dummies(pd.DataFrame([x]), columns=self.cat_cols)  # Apply dummy function
        missing = self._dummy_cols.difference(dummies.columns)  # Verify if there are missing columns
        dummies = dummies.assign(**{col: 0 for col in missing})  # Set missing columns to 0
        new_dummies_cols = dummies.columns.difference(x.index)  # Get columns names as dummies
        dummies[new_dummies_cols] = dummies[new_dummies_cols] * np.sqrt(.5)  # Replace categorical data by sqrt(0.5)
        return dummies.iloc[0]
