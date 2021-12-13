import logging
import warnings
from typing import Union, Type, Optional

import numpy as np
import pandas as pd
from kaog import KAOG

from kaogexp.data.loader import NOME_COLUNA_Y
from kaogexp.data.loader.DatasetAbstract import DatasetAbstract
from kaogexp.data.sampler.SamplerAbstract import SamplerAbstract
from kaogexp.explainer.methods.MethodAbstract import MethodAbstract
from kaogexp.model.ModelAbstract import ModelAbstract

warnings.filterwarnings('ignore', message=r'.*The feature names should match those.*', category=FutureWarning)


class KAOGExp:
    NUM_SAMPLES = 100
    LIMITE_EPSILON = 1

    def __init__(self, dataset: DatasetAbstract, modelo: ModelAbstract, sampler: SamplerAbstract):
        self.dataset = dataset
        self.modelo = modelo
        self.sampler = sampler

    def explicar(self,
                 instancia: Union[pd.Series, pd.DataFrame],
                 metodo: Type[MethodAbstract],
                 **kwargs,
                 ) -> Union[tuple[Optional[MethodAbstract], ...], MethodAbstract, None]:
        """
        Tem como objetivo explicar `instancia`, utilizando um método definido por `metodo`.
        Para isso, primeiramente é necessário fazer a verificação se os dados condizem com o esperado pelo modelo.
        Em seguida, é realizada a amostragem ao redor de `instancia` e a sua classificação usando o modelo. Para isso, é
        preciso que ela esteja no formato encode. Deve realizar a amostragem, até que encontre ao menos uma instância
        com a classificação desejada.
        Quando a amostragem for apropriada, então é criado o KAOG e enviado para `metodo`, que será responsável pela
        estratégia de encontrar a melhor explicação.

        :param instancia: Valor único ou conjunto de dados a serem explicados. Não deve estar tratado.
        :type instancia: Union[pd.Series, pd.DataFrame]
        :param metodo: Método a ser utilizado para explicar a instância. Deve ser apenas a referência a classe, ao invés do objeto já instânciado.
        :type metodo: MethodAbstract
        :param kwargs: Parâmetros adicionais para o método passado.
        :type kwargs: object
        :return: Explicação da instância.
        :rtype: Union[tuple[Optional[MethodAbstract], ...], MethodAbstract, None]
        :raise ValueError: Se o dado não estiver no tipo esperado.
        """
        self._assert_instance_compatibility(instancia)
        if isinstance(instancia, pd.DataFrame):
            return tuple(self._explicar(instancia, metodo, kwargs=kwargs) for _, instancia in instancia.iterrows())

        elif isinstance(instancia, pd.Series):
            return self._explicar(instancia, metodo, kwargs=kwargs)

        else:
            raise TypeError(f'Instance must be of type pd.Series or pd.DataFrame. Got {type(instancia)}.')

    def _explicar(self, instancia: pd.Series, metodo: Type[MethodAbstract], **kwargs) -> Optional[MethodAbstract]:
        """
        Lógica para a explicação
        """
        amostragem: Union[pd.Series, None] = None
        y_amostragem: Union[pd.Series, None] = None
        classe_desejada = kwargs.get('classe_desejada', None)
        try:
            while not self._amostra_valida(y_amostragem, classe_desejada):
                amostragem: pd.DataFrame = self._realizar_amostragem(instancia)
                y_amostragem: pd.Series = self._classificar_amostragem(amostragem)

            logging.info(f'Amostragem válida encontrada. Realizando KAOG.\n\n')
            amostragem_com_y = amostragem.copy()
            amostragem_com_y[NOME_COLUNA_Y] = y_amostragem
            amostra_completa = amostragem_com_y.append(instancia)
            kaog = self._criar_kaog(amostra_completa)
            return metodo(kaog, instancia, **kwargs)
        except ValueError:
            logging.info('Não foi possível encontrar uma amostra válida.\n\n')
            return None

    def _assert_instance_compatibility(self, instance: Union[pd.Series, pd.DataFrame]) -> None:
        """
        Compara as colunas da instância com as esperadas no tratador.

        :param instance: Dado que deve estar no mesmo formato dos dados de dataset.
        :type instance:  Union[pd.Series, pd.DataFrame]
        :rtype None
        :raise ValueError: Se o dado não estiver no tipo esperado.
        """
        if isinstance(instance, pd.Series):
            index = instance.index
        elif isinstance(instance, pd.DataFrame):
            index = instance.columns
        else:
            raise TypeError(f'Instance must be of type pd.Series or pd.DataFrame. Got {type(instance)}.')

        pd.testing.assert_index_equal(index, self.dataset.tratador.nomes_colunas_originais, check_order=False)

    def _amostra_valida(self, y_amostragem: Union[pd.Series, None], classe_desejada: int) -> bool:
        """
        Verifica se a amostragem é válida, ou seja, se existe alguma instância com a classificação desejada.
        Caso ainda não seja válida, incrementa o valor de epsilon e tenta novamente.

        :param y_amostragem: Valores da classificação da amostragem, ou None se ainda não foi definida.
        :type y_amostragem: Union[pd.Series, None]
        :param classe_desejada: Classe desejada.
        :type classe_desejada: int
        :return: True se a amostragem é válida, False caso contrário.
        :rtype: bool
        """
        if y_amostragem is None:
            return False

        desejada_any = (y_amostragem == classe_desejada).any()
        if not desejada_any:
            self._incrementar_epsilon()
        else:
            self._reset_epsilon()
        return desejada_any

    def _incrementar_epsilon(self) -> None:
        """
        Incrementa o valor de epsilon, caso a amostragem não seja válida.
        Se atingir o valor máximo, lança uma exceção.

        :raise ValueError: se o valor de epsilon atingir o valor máximo.
        """
        self.sampler.increase_epsilon()

    def _reset_epsilon(self):
        """
        Reseta o valor de epsilon para o valor inicial para a próxima amostragem.
        """
        self.sampler.reset_epsilon()

    def _realizar_amostragem(self, instancia: pd.Series) -> pd.DataFrame:
        """
        Dado um ponto de instância, realiza a amostragem ao redor dele.

        :param instancia: Ponto de instância a ser amostrado.
        :type instancia: Union[pd.Series, pd.DataFrame]
        :return: Amostragem ao redor do ponto de instância.
        :rtype: np.ndarray
        """
        logging.info(f'Realizando amostragem com epsilon {self.sampler.epsilon}.')
        if not isinstance(instancia, pd.Series):
            raise TypeError(f'`instancia` must be `pd.Series.` Got {type(instancia)}.')
        return self.sampler.realizar_amostragem(instancia, KAOGExp.NUM_SAMPLES)

    def _classificar_amostragem(self, amostragem: pd.DataFrame) -> pd.Series:
        """
        Dado o array de amostragem, classifica-o utilizando o modelo.

        :param amostragem: Amostragem a ser classificada.
        :type amostragem: np.ndarray
        :return: Classificação da amostragem.
        :rtype: np.ndarray
        """
        encoded = self.dataset.tratador.encode(amostragem)
        predict = self.modelo.predict(encoded)
        return pd.Series(predict, index=amostragem.index, name=NOME_COLUNA_Y)

    def _criar_kaog(self, amostra_completa: pd.DataFrame) -> KAOG:
        categorical_index = list(filter(lambda x: type(x) is int,
                                        [idx if col in self.dataset.nomes_colunas_categoricas else None for idx, col in
                                         enumerate(amostra_completa.columns)]))
        return KAOG(amostra_completa)
