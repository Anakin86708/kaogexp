import logging
import os
from os.path import join
from typing import Union

import numpy as np
import pandas as pd
import requests
import torch

from kaogexp.data.loader import ColunaYSingleton
from kaogexp.data.treatment.TreatmentAbstract import TreatmentAbstract
from kaogexp.model.ModelAbstract import ModelAbstract


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
        elif name == 'give_me_some_credit':
            url = 'https://github.com/carla-recourse/cf-models/raw/main/models/give_me_some_credit/ann.pt'
        elif name == 'compas':
            url = 'https://github.com/carla-recourse/cf-models/raw/main/models/compas/ann.pt'
        else:
            raise RuntimeError(f'Model {name} not found.')
        # Download file if not present
        path = join(os.path.dirname(__file__), 'models', f'{name}_ann.pt')
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
        x = x.copy()
        if isinstance(x, pd.Series):
            x = x.drop(ColunaYSingleton().NOME_COLUNA_Y, errors='ignore')
            return self._predict_series(x)
        elif isinstance(x, pd.DataFrame):
            x = x.drop(ColunaYSingleton().NOME_COLUNA_Y, axis=1, errors='ignore')
            return self._predict_dataframe(x)
        raise RuntimeError('x must be a pandas.Series or pandas.DataFrame')

    def _predict_dataframe(self, x: pd.DataFrame):
        try:
            tensor = torch.from_numpy(x.to_numpy(dtype=float))
            tensor = tensor.float()
            return self.raw_model(tensor)[:, 1].reshape((-1, 1)).round().detach().numpy().reshape(1, -1)[0]
        except ValueError:
            self.logger.debug('Applying treatment to dataframe.')
            return self._predict_dataframe(self.tratador.encode(x))

    def _predict_series(self, row: pd.Series):
        df = pd.DataFrame([row])
        return self._predict_dataframe(df)[0]
        # return self.raw_model(tensor)[:, 1].reshape(-1, 1)
