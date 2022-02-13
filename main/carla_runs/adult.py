# %%
# Carregar o adult dataset
import json
import logging
import operator

import pandas as pd

from kaogexp.data.loader import ColunaYSingleton
from kaogexp.model.ANN import ANN

ColunaYSingleton().NOME_COLUNA_Y = 'income'

from kaogexp.data.loader.DatasetFromMemory import DatasetFromMemory
from kaogexp.data.sampler.LatinSampler import LatinSampler
from kaogexp.explainer.KAOGExp import KAOGExp
from kaogexp.explainer.methods.Counterfactual import Counterfactual
from main.new_distance import NewDistance
from kaogexp.metrics.CERScore import CERScore
from kaogexp.metrics.carla_metrics import CARLADistances
from kaogexp.metrics.dispersao import Dispersao
from kaogexp.metrics.proximity import Proximity
from kaogexp.metrics.validity import Validity

logging.basicConfig(level=logging.DEBUG)

# cols_drop = ['fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'relationship', 'occupation',
#              'marital-status']
cols_drop = []
categorical_columns = ['workclass', 'marital-status', 'occupation', 'relationship',
                       'race', 'sex', 'native-country']
index = pd.Index(categorical_columns)
train_data = pd.read_csv('../../data/carla_data/adult_train.csv', index_col=0)
train_data.columns = train_data.columns.str.strip()
train_data = train_data.drop(cols_drop, axis=1)
train_data = DatasetFromMemory(train_data, index)

test_data = pd.read_csv('../../data/carla_data/adult_test.csv', index_col=0)
test_data.columns = test_data.columns.str.strip()
test_data = test_data.drop(cols_drop, axis=1)
test_data = DatasetFromMemory(test_data, index)

# %%
x = train_data.x(normalizado=True, encoded=True)
y = train_data.y()
model = ANN(train_data.tratador)
# %%
epsilon = 0.05
limite_epsilon = 1.0
seed = 42
sampler = LatinSampler(epsilon=epsilon, seed=seed, limite_epsilon=limite_epsilon)

# %%
metodo = Counterfactual
fixed_cols = pd.Index(['sex', 'age'])
explicador = KAOGExp(train_data, model, sampler, fixed_cols=fixed_cols, otimizar=True)
classe_desejada = 1
tratador_associado = train_data.tratador
normalizador_associado = train_data.normalizador

print('Realizando explicacao...')
explicacoes = explicador.explicar(test_data.dataset().sample(2), metodo=metodo, classe_desejada=classe_desejada,
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
# Unir todas as distancias de mesma métrica
cerscore_carla = {}
for distance in carla_distances[0].keys():
    op = operator.itemgetter(distance)
    cerscore_carla[distance] = cers.calcular(explicacoes, list(map(op, carla_distances)))

print(cerscore_carla)
metricas_dict = {
    'validade': {
        'proporcao_validade': (validades.count(True) / len(validades)),
    },
    'dispersao': dispersao,
    'proximidade': {
        'media_proximidade': pd.Series(proximidades).describe().to_dict(),
        'proximidades': proximidades
    },
    'cerscore': {
        'new_distance': cerscore,
        **cerscore_carla
    },
    'carla_distances': carla_distances,
}

with open('metricas_test.json', 'w') as f:
    json.dump(metricas_dict, f, indent=4)

# %%
# fig = Dispersao.plot(dispersao)
# fig.show()

# %%
cat_cols = test_data.nomes_colunas_categoricas
for i, item in enumerate(explicacoes):
    if item is not None and not item.instancia_original[cat_cols].equals(item.instancia_modificada[cat_cols]):
        print('Found ', i)
        break
