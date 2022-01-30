import logging
import os
from os.path import join
from typing import Union

import numpy as np
import pandas as pd
import requests
import torch

from data.treatment.TreatmentAbstract import TreatmentAbstract
from model.ModelAbstract import ModelAbstract


class ANN(ModelAbstract):
    """
    ANN model from PyTorch used in CARLA.
    More info at https://github.com/carla-recourse/cf-models
    """
    logger = logging.getLogger(__name__)

    def __init__(self, tratador: TreatmentAbstract, name: str = 'adult'):
        raw_model = self._get_model(name)
        super().__init__(raw_model, tratador)

    @staticmethod
    def _get_model(name: str):
        model = ANN._retrieve_model(name)
        model.eval()
        return model

    @staticmethod
    def _retrieve_model(name: str):
        """
        Load a pretrained model from GitHub.

        :return: PyTorch model
        """
        if name == 'adult':
            url = 'https://github.com/carla-recourse/cf-models/raw/main/models/adult/ann.pt'
        else:
            raise RuntimeError(f'Model {name} not found.')
        # Download file if not present
        path = join(os.path.dirname(__file__), 'models', 'ann.pt')
        if not os.path.isfile(path):
            ANN.logger.info('Model ANN is not present. Downloading..')
            r = requests.get(url)
            with open(path, 'wb') as model_file:
                if r.ok:
                    ANN.logger.info('Model ANN downloaded successfully.')
                    model_file.write(r.content)
        else:
            ANN.logger.info('Model ANN is already present.')

        return torch.load(path)

    def predict(self, x: Union[pd.Series, pd.DataFrame]) -> Union[int, np.ndarray]:
        if isinstance(x, pd.Series):
            return self._predict_series(x)
        elif isinstance(x, pd.DataFrame):
            return self._predict_dataframe(x)
        raise RuntimeError('x must be a pandas.Series or pandas.DataFrame')

    def _predict_dataframe(self, x: pd.DataFrame):
        tensor = torch.from_numpy(x.to_numpy(dtype=float))
        tensor = tensor.float()
        return self.raw_model(tensor)[:, 1].reshape((-1, 1)).round().detach().numpy()[0]

    def _predict_series(self, row: pd.Series):
        df = pd.DataFrame([row])
        return self._predict_dataframe(df)[0]
        # return self.raw_model(tensor)[:, 1].reshape(-1, 1)
