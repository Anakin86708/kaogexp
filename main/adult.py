# %%
# Carregar o adult dataset
import logging

import pandas as pd

from kaogexp.data.loader.DatasetFromMemory import DatasetFromMemory
from kaogexp.data.sampler.LatinSampler import LatinSampler
from kaogexp.explainer.KAOGExp import KAOGExp
from kaogexp.explainer.methods.Counterfactual import Counterfactual
from kaogexp.model.RandomForestModel import RandomForestModel
from main.new_distance import NewDistance
from metrics.CERScore import CERScore
from metrics.carla_metrics import CARLADistances
from metrics.dispersao import Dispersao
from metrics.proximity import Proximity
from metrics.validity import Validity

logging.basicConfig(level=logging.DEBUG)

categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                       'race', 'sex', 'native-country']
index = pd.Index(categorical_columns)
train_data = pd.concat([pd.read_csv('../data/adult/x_train.csv'), pd.read_csv('../data/adult/y_train.csv')], axis=1)
train_data.columns = train_data.columns.str.strip()
train_data = DatasetFromMemory(train_data, index)

test_data = pd.concat([pd.read_csv('../data/adult/x_test.csv'), pd.read_csv('../data/adult/y_test.csv')], axis=1)
test_data.columns = test_data.columns.str.strip()
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
fixed_cols = pd.Index(['sex', 'age', 'race', 'native-country'])
explicador = KAOGExp(train_data, model, sampler, fixed_cols=fixed_cols, otimizar=True)
classe_desejada = 1
tratador_associado = train_data.tratador
normalizador_associado = train_data.normalizador

print('Realizando explicacao...')
explicacoes = explicador.explicar(test_data.dataset().sample(1), metodo=metodo, classe_desejada=classe_desejada,
                                  tratador_associado=tratador_associado, normalizador_associado=normalizador_associado)

# %%
for item in explicacoes:
    try:
        print(item)
    except AttributeError:
        print('Empty')

# %%
# Métricas
logging.basicConfig(level=logging.INFO)
dist = NewDistance(test_data.dataset(), test_data.nomes_colunas_categoricas)
logging.basicConfig(level=None)
prox = Proximity(dist.calculate)
cers = CERScore(dist.calculate)
validades = []
dispersao = []
proximidades = []
carla_distances = []
for item in explicacoes:
    validades.append(Validity.calcular(item))
    dispersao.append(Dispersao.calcular(item))
    proximidades.append(prox.calcular(item))
    carla_distances.append(CARLADistances.calcular(item))
cerscore = cers.calcular(explicacoes, proximidades)

print('Validade:', validades)
print('Proporção de validade: %.3f' % (validades.count(True) / len(validades)))
print('Dispersão:', dispersao)
print('Proximidade:', proximidades)
print('Média:\n', pd.Series(proximidades).describe())
print('CERScore:', cerscore)
print('Carla Distances:')
for d in carla_distances:
    print(d)

# %%
# fig = Dispersao.plot(dispersao)
# fig.show()

# %%
cat_cols = test_data.nomes_colunas_categoricas
for i, item in enumerate(explicacoes):
    if item is not None and not item.instancia_original[cat_cols].equals(item.instancia_modificada[cat_cols]):
        print('Found ', i)
        break
