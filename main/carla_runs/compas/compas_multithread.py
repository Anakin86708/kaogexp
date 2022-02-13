# %%
# Carregar o adult dataset
import json
import logging
import multiprocessing
import operator
import os
import pickle
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd

from kaogexp.data.loader import ColunaYSingleton
from kaogexp.model.ANN import ANN
from main.carla_runs.util import save_tratador_and_normalizador

ColunaYSingleton().NOME_COLUNA_Y = 'score'

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

logging.basicConfig(level=logging.INFO)

name = 'compas'
categorical_columns = ['c_charge_degree', 'race', 'sex']
index = pd.Index(categorical_columns)
train_data = pd.read_csv('../../../data/carla_data/compas_train.csv', index_col=0)
train_data.columns = train_data.columns.str.strip()
train_data = DatasetFromMemory(train_data, index)

test_data = pd.read_csv('../../../data/carla_data/compas_test.csv', index_col=0)
test_data.columns = test_data.columns.str.strip()
test_data = DatasetFromMemory(test_data, index)

# %%
x = train_data.x(normalizado=True, encoded=True)
y = train_data.y()
model = ANN(train_data.tratador, name=name)

# %%
epsilon = 0.05
limite_epsilon = 1.0
seed = 42
sampler = LatinSampler(epsilon=epsilon, seed=seed, limite_epsilon=limite_epsilon)

# %%
metodo = Counterfactual
fixed_cols = pd.Index(['age', 'race', 'sex'])
classe_desejada = 1
tratador_associado = train_data.tratador
normalizador_associado = train_data.normalizador

working_dir = os.path.dirname(__file__)
save_tratador_and_normalizador(working_dir, name, tratador_associado, normalizador_associado)

print('Realizando explicacao...')
threads = []
threads_num = multiprocessing.cpu_count()


def explicar(item, i, total):
    logging.info('\n' + ('#' * 15) + f' Item {i + 1} of {total} ' + ('#' * 15) + '\n')
    explicador = KAOGExp(train_data, model, sampler, fixed_cols=fixed_cols, otimizar=True)
    return explicador.explicar(item, metodo=metodo, classe_desejada=classe_desejada,
                               tratador_associado=tratador_associado, normalizador_associado=normalizador_associado)


# Quantidade de instâncias que serão explicadas do conjunto de testes
NUM_SAMPLE_DATASET = 100
test_dataset = test_data.dataset().sample(NUM_SAMPLE_DATASET, random_state=seed)
with ThreadPoolExecutor(max_workers=threads_num) as executor:
    total = len(test_dataset)
    for i, (idx, row) in enumerate(test_dataset.iterrows()):
        threads.append(executor.submit(explicar, row, i, total))

explicacoes = tuple(map(lambda th: th.result(), threads))

# %%
with open(os.path.join('pkls', f'{name}.pkl'), 'wb') as file:
    for item in explicacoes:
        try:
            pickle.dump(item, file)
        except AttributeError:
            print('Empty')

# %%
# Métricas
logging.info("Computando métricas")
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
    try:
        carla_distances.append(CARLADistances.calcular(item))
    except AttributeError:
        pass

cerscore = cers.calcular(explicacoes, proximidades)

# Unir todas as distancias de mesma métrica
cerscore_carla = {}
for distance in carla_distances[0].keys():
    op = operator.itemgetter(distance)
    cerscore_carla[distance] = cers.calcular(explicacoes, list(map(op, carla_distances)))

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

with open(f'metricas_{name}.json', 'w') as f:
    json.dump(metricas_dict, f, indent=4)

logging.info("Fim")
# %%
# fig = Dispersao.plot(dispersao)
# fig.show()

# %%
# Verificar se está tendo alterações em dados categóricos
cat_cols = test_data.nomes_colunas_categoricas
for i, item in enumerate(explicacoes):
    if item is not None and not item.instancia_original[cat_cols].equals(item.instancia_modificada[cat_cols]):
        print('Found ', i)
