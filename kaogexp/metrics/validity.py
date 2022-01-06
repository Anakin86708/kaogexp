from kaogexp.explainer.methods.Counterfactual import Counterfactual


class Validity():
    """Compara se a classe desejada foi atingida ou não."""

    Explicadores = Counterfactual

    def __init__(self):
        pass

    @staticmethod
    def calcular(instancia: Explicadores) -> bool:
        """ Calcula a validade da classe de acordo com certa instância
        """
        try:
            return Validity._e_igual_classe_desejada(instancia)
        except:
            return False

    @staticmethod
    def _e_igual_classe_desejada(instancia):
        return instancia.classe_modificada == instancia.classe_desejada
