from copy import deepcopy
from itertools import combinations
from typing import List

import pandas as pd

from data.loader import ColunaYSingleton
from explainer.methods.Counterfactual import Counterfactual
from kaogexp.explainer.methods.MethodAbstract import MethodAbstract
from kaogexp.model.ModelAbstract import ModelAbstract
from metrics.dispersao import Dispersao


class SparsityOptimization:
    """ Redução do número de features alteradas pelas buscas
    Consiste em tentar reverter features que tiveram pouca alteração para o valor
    anterior e ainda assim manter uma classificação.
    """

    def __init__(self, modelo: ModelAbstract, cat_cols: pd.Index):
        self.modelo = modelo
        self.cat_cols = cat_cols.copy()
        self._instancia_original = None

    def optimize(self, instancia: Counterfactual):
        """
        Otimiza a instância, tentando reverter as features para o original. A otimização é aceita, se a classe for
        mantida.

        :param instancia: Instância a ser otimizada.
        :type instancia: MethodAbstract
        :return: Instância otimizada, podendo ser a mesma que a entrada, se não houve mudança.
        :rtype: MethodAbstract
        """
        instancia = deepcopy(instancia)
        self._instancia_original: pd.Series = instancia.instancia_original
        instancia_modificada: pd.Series = instancia.instancia_modificada
        instancia_otimizada: pd.Series = instancia_modificada.copy()

        # Obter pemutacoes de features
        for item in self._permutar_features(instancia):
            # Classificar os alterados
            # se encontrar algum que permanceça na classe desejada, parar
            classe_otimizada = self.modelo.predict(item)
            if classe_otimizada == instancia.classe_desejada:
                item[ColunaYSingleton().NOME_COLUNA_Y] = classe_otimizada
                instancia._instancia_modificada = item
                break
        return instancia

    def _permutar_features(self, instancia: Counterfactual):
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
        metrica = Dispersao.calcular(instancia)
        original = instancia.instancia_original
        modificada = instancia.instancia_modificada
        max_can_change = metrica - 1
        for num_alterados in range(max_can_change, 0, -1):
            idx_alterados = original.index[original != modificada]
            poss_idx_alterar = list(map(pd.Index, combinations(idx_alterados, num_alterados)))
            for item in poss_idx_alterar:
                ins = modificada.copy()
                ins[item] = original[item]
                yield ins.copy()
