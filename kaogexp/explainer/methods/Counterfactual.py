import pandas as pd
from kaog import KAOG

from kaogexp.explainer.methods.MethodAbstract import MethodAbstract


class Counterfactual(MethodAbstract):

    def __init__(self, kaog: KAOG, instancia_buscada: pd.Series):
        super().__init__(kaog, instancia_buscada)
        self._realizar_busca()

    @property
    def instancia_modificada(self) -> pd.Series:
        return self._instancia_modificada.copy()

    def _realizar_busca(self) -> pd.Series:
        """
        Utilizando o KAOG, devem ser encontrados os valores mais próximos da instância procurada.
        Iterando os valores mais próximos, deve ser selecionado aquele que satisfazer a condição de busca.
        :return: Instância resultado da busca.
        :rtype: pd.Series
        """
        distancias = self.kaog.distancias
        # TODO: Implementar a busca
