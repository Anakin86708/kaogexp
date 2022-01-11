from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from kaogexp.data.loader.DatasetAbstract import DatasetAbstract
from kaogexp.model.ModelAbstract import ModelAbstract


class RandomForestModel(ModelAbstract):

    def __init__(self, x: pd.DataFrame, y: pd.Series):
        model = RandomForestClassifier()
        model.fit(x, y)
        super().__init__(model)

    @classmethod
    def from_dataset(cls, dataset: DatasetAbstract):
        """
        Creates a new RandomForestModel from a DatasetAbstract, using the data to train the model.

        :param dataset: Data to be used to train the model.
        :type dataset: DatasetAbstract
        :return: Objeto RandomForestModel
        :rtype: RandomForestModel
        """
        x = dataset.x(True, True)
        y = dataset.y()
        return cls(x, y)

    def predict(self, x: Union[pd.Series, pd.DataFrame]) -> Union[int, np.ndarray]:
        if isinstance(x, pd.Series):
            predict = self.raw_model.predict(x.values.reshape(1, -1))[0]
        else:
            predict = self.raw_model.predict(x)
        return predict
