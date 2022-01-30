from kaog.util import ColunaYSingleton as CYS


class ColunaYSingleton(CYS):

    @CYS.NOME_COLUNA_Y.setter
    def NOME_COLUNA_Y(self, nome):
        CYS().NOME_COLUNA_Y = nome
        self._nome_coluna_y = nome
