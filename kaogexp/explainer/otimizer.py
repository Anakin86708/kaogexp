from typing import List

import pandas as pd

from kaogexp.explainer.methods.MethodAbstract import MethodAbstract
from kaogexp.model.ModelAbstract import ModelAbstract


class SparsityOptimization:
    """ Redução do número de features alteradas pelas buscas
    Consiste em tentar reverter features que tiveram pouca alteração para o valor
    anterior e ainda assim manter uma classificação.
    """

    def __init__(self, modelo: ModelAbstract, cat_cols: pd.Index):
        self.modelo = modelo
        self.cat_cols = cat_cols.copy()
        self._instancia_original = None

    def optimize(self, instancia: MethodAbstract):
        """
        Otimiza a instância, tentando reverter as features para o original. A otimização é aceita, se a classe for
        mantida.

        :param instancia: Instância a ser otimizada.
        :type instancia: MethodAbstract
        :return: Instância otimizada, podendo ser a mesma que a entrada, se não houve mudança.
        :rtype: MethodAbstract
        """
        self._instancia_original: pd.Series = instancia.instancia_original
        instancia_modificada: pd.Series = instancia.instancia_modificada
        instancia_otimizada: pd.Series = instancia_modificada.copy()

        # Obter pemutacoes de features

    def _permutar_features(self, instancia: pd.Series, changed: List, remaining: List):
        permutacoes = [instancia.copy()]
        intalteradas = []
        if len(remaining) == 0:
            return permutacoes

        for i, item in enumerate(remaining):
            alterada = instancia.copy()
            if alterada[item] != self._instancia_original[item]:
                alterada[item] = self._instancia_original[item]
                rem = list(filter(lambda x: x != item and x not in intalteradas, remaining))
                permutacoes.extend(self._permutar_features(alterada, changed + [item], rem))
            else:
                intalteradas.append(item)
        return permutacoes
