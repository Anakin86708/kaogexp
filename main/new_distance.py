import numpy as np
import pandas as pd


class NewDistance:

    def __init__(self, data: pd.DataFrame, cat_cols: pd.Index):
        self._data = data.copy()
        self._cat_cols = cat_cols.copy()
        self._range = self._get_range()

    @property
    def data(self):
        return self._data.copy()

    @property
    def range(self):
        return self._range.copy()

    @property
    def cat_cols(self):
        return self._cat_cols.copy()

    def calculate(self, x, y):
        """
        Calculate the normalized eucliian distance between two series, considering the categorical data distance as
        1 if diferent or 0 if equal.

        :param x: Series with the first data.
        :param y: Series with the second data.
        :return: The normalized eucliian distance between the two series.
        """
        if not self.cat_cols.empty:
            x_, y_ = self._apply_dummy(pd.Series(x)), self._apply_dummy(pd.Series(y))
        else:
            x_, y_ = x, y
        return np.sqrt(np.sum(np.square(np.divide((x_ - y_), self._range))))

    def _get_range(self):
        """Get the range of the data.
        Max and min values are calculated for each column.
        Categorical colmuns are ignored and get the value 1.

        :return: Series with the range
        :rtype: pd.Series
        """
        range_ = abs(self._data.max() - self._data.min())
        range_[self._cat_cols] = 1
        return range_

    def _apply_dummy(self, x: pd.Series) -> pd.Series:
        """Apply the dummy function to the series.
        All the categorical data are replaced by sqrt(0.5), and the rest by 0. That allows to calculate the square and
        sum the values to obtain 1 in case of diferent values.

        :param x: Series to be applied.
        :type x: pd.Series
        :return: Series with one-hot encoding.
        :rtype: pd.Series
        """
        dummies = pd.get_dummies(x, columns=self._cat_cols)
        dummies[self._cat_cols] = dummies[self._cat_cols] * np.sqrt(.5)
        return dummies
