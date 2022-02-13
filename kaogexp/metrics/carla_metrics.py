import numpy as np

from kaogexp.explainer.methods.Counterfactual import Counterfactual


class CARLADistances:

    @staticmethod
    def calcular(instancia: Counterfactual):
        if not isinstance(instancia, Counterfactual):
            return None

        results = {}
        tratador = instancia.tratador_associado
        delta = tratador.encode(instancia.instancia_modificada) - tratador.encode(instancia.instancia_original)
        results['Distance_1'] = CARLADistances._d1(delta)
        results['Distance_2'] = CARLADistances._d2(delta)
        results['Distance_3'] = CARLADistances._d3(delta)
        results['Distance_4'] = CARLADistances._d4(delta)

        return results

    @staticmethod
    def _d1(delta):
        return float(np.sum(delta != 0))

    @staticmethod
    def _d2(delta):
        """SAD"""
        return float(np.sum(np.abs(delta)))

    @staticmethod
    def _d3(delta):
        """SSD"""
        return float(np.sum(np.square(np.abs(delta))))

    @staticmethod
    def _d4(delta):
        return float(np.max(np.abs(delta)))
