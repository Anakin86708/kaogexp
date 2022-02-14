import logging
from typing import List, Union, Callable, Iterable

import numpy as np

from kaogexp.explainer.methods.MethodAbstract import MethodAbstract
from model.ModelAbstract import ModelAbstract


class CERScore:
    """
    Métrica de robustez para modelos caixa-preta.

    Referência
    -----------------
    S. Sharma, J. Henderson, and J. Ghosh, “CERTIFAI: A common frame-work  to  provide  explanations  and  analyse  the  fairness  and  robustnessof black-box models,” inProceedings of the Twenty Third InternationalConference on Artificial Intelligence and Statistics, ser. Proceedings ofthe 2020 AAAI/ACM Conference on AI, Ethics, and Society, S. Chiappaand R. Calandra, Eds., vol. 108.    ACM, 2020, pp. 166–172
    """

    logger = logging.getLogger(__name__)

    def __init__(self, metodo_distancias: Callable, model: ModelAbstract = None):
        self.metodo_distancias = metodo_distancias
        self.model = model

    def calcular(self, x: Iterable[MethodAbstract], distancias: List[float] = None) -> Union[
        float, None]:
        """
        Cálculo da métrica CERScore definidade pela Eq. 9.

        :param x: Lista contendo as instâncias de Contrastive ou Counterfactual.
        :param distancias: Lista que corresponde a distância de entre o ponto de interesse e a instância encontrada de cada item de `x`.
        :return: Valor esperado das distâncias.
        """
        return self._expected(x, distancias)

    def _expected(self, x: Iterable[MethodAbstract], distancias: List[float]) -> Union[
        float, None]:
        """
        Deve ser calculada a distância de todas as instâncias, se não tiver sido feito, e retornado o valor esperado.
        Como não há diferentes probabilidades, é retornado apenas a média.

        :param x: Lista contendo as instâncias de Contrastive ou Counterfactual.
        :param distancias: Lista que corresponde a distância de entre o ponto de interesse e a instância encontrada de cada item de `x`.
        :return: Valor da média das distâncias se possível, senão, None.
        """
        try:
            if distancias is None:
                distancias = [self._calcular_distancia(instancia) for instancia in x]

            filtred_distances = filter(lambda item: item is not None, distancias)
            try:
                self.logger.debug(f'Trying to calculate with prob')
                prob = np.array([self.model.prob(i.instancia_modificada) for i in x])
                return sum(prob * distancias)
            except Exception:
                self.logger.debug(f'Trying to calculate without prob')
                return self._mean(distancias, filtred_distances)
        except Exception:
            self.logger.debug(f'CERScore failed')
            return None

    def _mean(self, distancias, filtred_distances):
        """
        Calcula a média das distâncias, levando em conta até mesmo os valores None.

        :param distancias: Lista com todas as distâncias, mesmo aquelas que não puderam ser calculadas.
        :type distancias: List[float]
        :param filtred_distances: Lista com as distâncias que puderam ser calculadas.
        :return: Média das distâncias.
        """
        return sum(filtred_distances) / len(distancias)

    def _calcular_distancia(self, instancia: MethodAbstract):
        return self.metodo_distancias(instancia.instancia_original, instancia.instancia_modificada)
