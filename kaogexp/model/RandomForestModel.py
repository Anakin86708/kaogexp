import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from kaogexp.data.loader import ColunaYSingleton
from kaogexp.data.loader.DatasetAbstract import DatasetAbstract
from kaogexp.data.treatment.TreatmentAbstract import TreatmentAbstract
from kaogexp.model.ModelAbstract import ModelAbstract


class RandomForestModel(ModelAbstract):
    __logger = logging.getLogger(__name__)

    def __init__(self, x: pd.DataFrame, y: pd.Series, tratador: TreatmentAbstract):
        model = RandomForestClassifier()
        model.fit(x, y)
        super().__init__(model, tratador)

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
        return cls(x, y, dataset.tratador)

    def predict(self, x: Union[pd.Series, pd.DataFrame]) -> Union[int, np.ndarray]:
        if isinstance(x, pd.Series):
            predict = self._predict_series(x)
        else:
            predict = self._predict_dataframe(x)
        return predict

    def _predict_dataframe(self, x: pd.DataFrame):
        x.drop(ColunaYSingleton().NOME_COLUNA_Y, errors='ignore', axis=1, inplace=True)
        if x.shape[1] == self.raw_model.n_features_in_:
            self.__logger.debug('Data already encoded')
            predict = self.raw_model.predict(x)
        else:
            self.__logger.debug('Encoding data for predict')
            predict = self.raw_model.predict(self.encode(x))
        return predict

    def _predict_series(self, x: pd.Series):
        x.drop(ColunaYSingleton().NOME_COLUNA_Y, errors='ignore', inplace=True)
        if x.shape[0] == self.raw_model.n_features_in_:
            self.__logger.debug('Series predict already encoded')
            x_values_reshape = x.values.reshape(1, -1)
            predict = self.raw_model.predict(x_values_reshape)[0]
        else:
            self.__logger.debug('Encoding series for predict')
            predict = self.raw_model.predict(self.tratador.encode(x).values.reshape(1, -1))[0]
        return predict
