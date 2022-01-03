import numpy as np
import pandas as pd


class NewDistance:

    def __init__(self, data: pd.DataFrame, cat_cols: pd.Index):
        self._data = data.copy()
        self._cat_cols = cat_cols.copy()
        self._range = self._get_range()
        self._dummy_cols = pd.get_dummies(data, columns=cat_cols).columns

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
        if isinstance(x, np.ndarray):
            x = pd.Series(x, index=self.data.columns)
        if isinstance(y, np.ndarray):
            y = pd.Series(y, index=self.data.columns)

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
        return pd.get_dummies(self.data, columns=self.cat_cols).apply(
            lambda x: 1 if x.name in self.cat_cols else abs(x.max() - x.min()))

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
