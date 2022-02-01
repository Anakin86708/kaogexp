import numpy as np

from kaogexp.explainer.methods.Counterfactual import Counterfactual


class CARLADistances:

    @staticmethod
    def calcular(instancia: Counterfactual):
        results = {}
        tratador = instancia.tratador_associado
        delta = tratador.encode(instancia.instancia_modificada) - tratador.encode(instancia.instancia_original)
        results['d1'] = CARLADistances._d1(delta)
        results['d2'] = CARLADistances._d2(delta)
        results['d3'] = CARLADistances._d3(delta)
        results['d4'] = CARLADistances._d4(delta)

        return results

    @staticmethod
    def _d1(delta):
        return np.sum(delta != 0)

    @staticmethod
    def _d2(delta):
        return np.sum(np.abs(delta))

    @staticmethod
    def _d3(delta):
        return np.sum(np.square(np.abs(delta)))

    @staticmethod
    def _d4(delta):
        return np.max(np.abs(delta))
