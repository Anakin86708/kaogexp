from typing import List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from data.loader import ColunaYSingleton
from kaogexp.explainer.methods.Counterfactual import Counterfactual


class Dispersao():
    """ Calcula quantas features foram alteradas - Dispersão
    """

    Explicadores = Counterfactual

    def __init__(self):
        pass

    @staticmethod
    def calcular(instancia: Explicadores):
        """
        Calcula a quantidade de features alteradas em `instancia`, comparando o ponto de interesse
        com o valor encontrado.
        """
        if instancia is None:
            return None
        return Dispersao._calcular_diferenca_features(instancia)

    @staticmethod
    def _calcular_diferenca_features(instancia: Explicadores):
        """
        Retorna a quantidade de features que foram alteradas em `instancia` com relação aos dados originais.

        :param instancia: Dados a serem comparados.
        :type instancia: Explicadores
        :return: Quantidade de features alteradas.
        :rtype: int
        """
        return np.count_nonzero(
            instancia.instancia_original.drop(
                [ColunaYSingleton().NOME_COLUNA_Y]).to_numpy() != instancia.instancia_modificada.drop(
                [ColunaYSingleton().NOME_COLUNA_Y]).to_numpy())

    @staticmethod
    def plot(dispersao: List[int]):
        """
        Realiza o plot com a distribuição da dispersão calculada.
        Parameters
        ----------
        dispersao: List[int]
            Valores de dispersão calculados.
        Returns
        -------
        plt.Figure
            Figura com o gráfico representando a dispersão.
        """
        dispersao = list(filter(lambda x: x is not None, dispersao))
        fig: plt.Figure = sns.displot(dispersao, kde=True, discrete=True, shrink=.9).figure
        plt.title('Dispersão')
        try:
            plt.xticks(list(range(min(dispersao), max(dispersao) + 1)))
        except:
            pass
        return fig
