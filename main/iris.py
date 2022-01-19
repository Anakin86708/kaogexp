# %%
# Carregar os dados padronizados de treino e teste do Iris
import logging

import pandas as pd

from kaogexp.data.loader.DatasetFromMemory import DatasetFromMemory
from kaogexp.data.sampler.LatinSampler import LatinSampler
from kaogexp.explainer.KAOGExp import KAOGExp
from kaogexp.explainer.methods.Counterfactual import Counterfactual
from kaogexp.metrics.CERScore import CERScore
from kaogexp.metrics.dispersao import Dispersao
from kaogexp.metrics.proximity import Proximity
from kaogexp.metrics.validity import Validity
from kaogexp.model.RandomForestModel import RandomForestModel

logging.basicConfig(level=logging.DEBUG)

categorical_columns = []
index = pd.Index(categorical_columns)
train_data = pd.concat([pd.read_csv('../data/iris/x_train.csv'), pd.read_csv('../data/iris/y_train.csv')], axis=1)
train_data = DatasetFromMemory(train_data, index)

test_data = pd.concat([pd.read_csv('../data/iris/x_test.csv'), pd.read_csv('../data/iris/y_test.csv')], axis=1)
test_data = DatasetFromMemory(test_data, index)

# %%
x = train_data.x(normalizado=True, encoded=True)
y = train_data.y()
model = RandomForestModel(x, y, train_data.tratador)

# %%
epsilon = 0.05
limite_epsilon = 1.0
seed = 42
sampler = LatinSampler(epsilon=epsilon, seed=seed, limite_epsilon=limite_epsilon)
# %%
metodo = Counterfactual
explicador = KAOGExp(train_data, model, sampler)
classe_desejada = 2
tratador_associado = train_data.tratador
normalizador_associado = train_data.normalizador

print('Realizando explicacao...')
explicacao = explicador.explicar(test_data.dataset().sample(2), metodo=metodo, classe_desejada=classe_desejada,
                                 tratador_associado=tratador_associado, normalizador_associado=normalizador_associado)

# %%
for item in explicacao:
    try:
        print(item)
    except AttributeError:
        print('Empty')

# %%
# Métricas
logging.basicConfig(level=None)
prox = Proximity(dist.calculate)
cers = CERScore(dist.calculate)
validades = []
dispersao = []
proximidades = []
for item in explicacao:
    validades.append(Validity.calcular(item))
    dispersao.append(Dispersao.calcular(item))
    proximidades.append(prox.calcular(item))
cerscore = cers.calcular(explicacao, proximidades)

print('Validade:', validades)
print('Proporção de validade: %.3f' % (validades.count(True) / len(validades)))
print('Dispersão:', dispersao)
fig = Dispersao.plot(dispersao)
fig.show()
print('Proximidade:', proximidades)
print('Média:\n', pd.Series(proximidades).describe())
print('CERScore:', cerscore)
