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
        """
        Realiza a permutação de features, revertendo as features que tiveram alguma alteração para o valor original.

        :param instancia: Instância que foi alterada.
        :type instancia: pd.Series
        :param changed: Lista com os nomes das colunas que foram alteradas.
        :type changed: List
        :param remaining: Lista com os nomes das colunas que não foram alteradas ainda.
        :type remaining: List
        :return: Todas as possíveis permutações de features.
        :rtype: pd.DataFrame
        """
        permutacoes = pd.DataFrame([instancia.copy()])
        intalteradas = []
        if len(remaining) == 0:
            return permutacoes

        for i, item in enumerate(remaining):
            alterada = instancia.copy()
            if alterada[item] != self._instancia_original[item]:
                alterada[item] = self._instancia_original[item]
                rem = list(filter(lambda x: x != item and x not in intalteradas, remaining))
                permutacoes = permutacoes.append(self._permutar_features(alterada, changed + [item], rem),
                                                 ignore_index=True)
            else:
                intalteradas.append(item)
        return permutacoes.drop_duplicates()
