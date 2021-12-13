import numpy as np
import pandas as pd
from kaog import KAOG

from kaogexp.explainer.methods.MethodAbstract import MethodAbstract


class Counterfactual(MethodAbstract):

    def __init__(self, kaog: KAOG, instancia_explicada: pd.Series, classe_desejada: int):
        super().__init__(kaog, instancia_explicada)
        self._classe_desejada = classe_desejada

        self.distancias_e_vizinhos = self.kaog.distancias_e_vizinhos
        self._instancia_modificada = self._realizar_busca()

    @property
    def instancia_modificada(self) -> pd.Series:
        return self._instancia_modificada.copy()

    @property
    def index_buscado(self):
        return self.instancia_original.name

    @property
    def classe_desejada(self):
        return self._classe_desejada

    @staticmethod
    def set_metrica_distancia(metrica):
        """
        Define a métrica de distância a ser utilizada para o KAOG.
        Pode ser definida como uma função ou como um nome de métrica reconhecida pelo
        NearestNeighbors

        :param metrica:
        :return:
        """
        KAOG.set_metrica_distancia(metrica)

    def _realizar_busca(self) -> pd.Series:
        """
        Utilizando o KAOG, devem ser encontrados os valores mais próximos da instância procurada.
        Iterando os valores mais próximos, deve ser selecionado aquele que satisfazer a condição de busca.

        :return: Instância resultado da busca.
        :rtype: pd.Series
        """
        vizinhos = self._obter_vizinhos()
        for vizinho in vizinhos:
            if self._condicao_busca(vizinho):
                return self._obter_instancia_modificada(vizinho).copy()

    def _obter_instancia_modificada(self, index: int):
        """
        Obter uma instância a partir dos dados do KAOG que satisfaz a condição de busca.

        :param index: Index da instância encontrada.
        :type index: int
        :return: Instância encontrada.
        """
        return self.kaog.data.loc[index]

    def _obter_vizinhos(self):
        """
        Obtém os vizinhos mais próximos da instância procurada.

        :return: Array contendo todos os vizinhos mais próximos da instância procurada.
        :rtype: np.ndarray
        """
        dist = self.distancias_e_vizinhos
        k = self.kaog.x.shape[0] - 1
        return dist.k_vizinhos_mais_proximos_de(k, self.index_buscado)

    def _condicao_busca(self, index_buscado: int):
        """
        Verifica se a instância procurada satisfaz a condição de busca.
        Para isso, é preciso que esteja em uma classe igual a buscada e que a pureza do instância encontrada seja maior
        que a atual.

        :param index_buscado: Índice da instância procurada.
        :type index_buscado: int
        :return: True se satisfazer a condição de busca.
        :rtype: bool
        """
        return self._e_classe_buscada(index_buscado) and self._e_maior_pureza(index_buscado)

    def _e_classe_buscada(self, index_buscado: int):
        """
        Verifica se a instância encontrada é de uma classe igual da classe procurada.

        :param index_buscado: Índice da instância procurada.
        :type index_buscado: int
        :return: True se satisfazer a condição da classe.
        :rtype: bool
        """
        classe_encontrada = self.kaog.y.loc[index_buscado]
        return self.classe_desejada == classe_encontrada

    def _e_maior_pureza(self, index_buscado: int):
        """
        Verifica se a pureza do componente ao qual a instância encontrada pertence é **maior ou igual** que a pureza do componente ao
        qual a instância procurada pertence.

        :param index_buscado: Índice da instância procurada.
        :type index_buscado: int
        :return: True se satisfazer a condição da pureza.
        :rtype: bool
        """
        pureza_original = self.kaog.grafo_otimo.pureza(self.instancia_original.name)
        pureza_encontrada = self.kaog.grafo_otimo.pureza(index_buscado)
        return pureza_encontrada >= pureza_original
