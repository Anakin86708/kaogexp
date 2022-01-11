from typing import Callable

from kaogexp.explainer.methods.Counterfactual import Counterfactual


class Proximity:
    """ Calcula a distância entre a instância original e o ponto encontrado
    Por padrão, é utilizado o modelo HEOM para calcular a distância entre
    as instâncias.
    """

    def __init__(self, metodo_distancia: Callable):
        self.metodo_distancia = metodo_distancia

    def calcular(self, instancia: Counterfactual):
        if instancia is None:
            return None
        return self.metodo_distancia(instancia.instancia_original, instancia.instancia_modificada)
