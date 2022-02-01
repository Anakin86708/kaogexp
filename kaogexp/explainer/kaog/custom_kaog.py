import logging
from math import sqrt

import numpy as np
import pandas as pd
from kaog import KAOG
from kaog.distancias import Distancias
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from kaogexp.data.loader.DatasetAbstract import DatasetAbstract


class KAOGAdaptado(KAOG):

    def __init__(self, dataset: DatasetAbstract):
        """
        Permite o uso de um dataset customizado para a criação do KAOG.

        :param dataset: Representação do dataset.
        :type dataset: DatasetAbstract
        """
        self.dataset = dataset

        data = dataset.dataset(normalizado=True, encoded=False)
        cat_cols = dataset.nomes_colunas_categoricas
        super().__init__(data, cat_cols)

    def _calcular_distancias_e_vizinhos(self):
        self._dist = NovaDistancia(self.dataset)

    def _get_tsne(self):
        tsne = TSNE(init='pca', learning_rate='auto', n_jobs=-1)
        tsne_cords = pd.DataFrame(tsne.fit_transform(self.dataset.x(factorized=True)), index=self.x.index)
        return tsne_cords


class NovaDistancia(Distancias):

    def __init__(self, dataset: DatasetAbstract):
        """
        Permite o uso de um dataset customizado para o cálculo das distâncias.

        :param dataset:
        """
        self.dataset = dataset
        x = dataset.x(normalizado=True, encoded=False)
        cat_cols = dataset.nomes_colunas_categoricas
        super().__init__(x, cat_cols)

    def _calcular_distancias_e_vizinhos(self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        k = data.shape[0]
        logging.debug('Calculando distâncias e vizinhos...')
        x = self._tratar_x(data)
        nn = NearestNeighbors(n_neighbors=k, metric=self.METRIC, n_jobs=1, algorithm='ball_tree').fit(x)

        # Decrementa k para considerar o próprio ponto
        distances, kneighbors = nn.kneighbors(n_neighbors=k - 1, return_distance=True)
        logging.debug('Distâncias calculadas.')
        self._ordenar(distances, kneighbors)
        return distances, kneighbors

    def _tratar_x(self, x) -> pd.DataFrame:
        """
        Valores de x devem ser colocados no formato de encode, e em seguida, substituidos as colunas de encode com valores
        1 para sqrt(0.5), para que possa ser utilizada a distância euclidiana nos dados categóricos.

        :param x: Dados para o cálculo das distâncias. Recebido em formato sem encode, com dados categórios em str.
        Não deve conter informação de classe.
        :type x: pd.DataFrame
        :return: Dados em formato encoded e valores sqrt(0.5) para os dados categóricos ao invés de 1.
        :rtype pd.DataFrame
        """
        tratador = self.dataset.tratador
        x_encoded = tratador.encode(x)
        colunas_encoded = tratador.nomes_colunas_categoricas_encoded
        x_encoded[colunas_encoded] = x_encoded[colunas_encoded].replace(1, sqrt(0.5))
        return x_encoded
