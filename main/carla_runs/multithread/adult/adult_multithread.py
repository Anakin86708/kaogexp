# %%
# Carregar o adult dataset
import logging
import multiprocessing
import pickle
from concurrent.futures.thread import ThreadPoolExecutor

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

logging.basicConfig(level=logging.INFO)

categorical_columns = ['workclass', 'marital-status', 'occupation', 'relationship',
                       'race', 'sex', 'native-country']
index = pd.Index(categorical_columns)
train_data = pd.read_csv('../../../../data/carla_data/adult_train.csv', index_col=0)
train_data.columns = train_data.columns.str.strip()
train_data = DatasetFromMemory(train_data, index)

test_data = pd.read_csv('../../../../data/carla_data/adult_test.csv', index_col=0)
test_data.columns = test_data.columns.str.strip()
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
classe_desejada = 1
tratador_associado = train_data.tratador
normalizador_associado = train_data.normalizador

with open('adult_tratador.pkl', 'wb') as file:
    pickle.dump(tratador_associado, file)

with open('adult_normalizador.pkl', 'wb') as file:
    pickle.dump(normalizador_associado, file)

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
with open('adult.pkl', 'wb') as file:
    for item in explicacoes:
        try:
            pickle.dump(item, file)
            print(item)
        except AttributeError:
            print('Empty')

# %%
# Métricas
logging.info("Computando métricas")
logging.basicConfig(level=logging.INFO)
dist = NewDistance(test_dataset, test_data.nomes_colunas_categoricas)
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

with open('adult.result', 'w', encoding='utf8') as file:
    file.write(f'Validade: {validades}\n')
    file.write(f'Proporção de validade: %.3f\n' % (validades.count(True) / len(validades)))
    file.write(f'Dispersão: {dispersao}\n')
    file.write(f'Proximidade: {proximidades}\n')
    file.write(f'Média:\n{pd.Series(proximidades).describe()}\n')
    file.write(f'CERScore: {cerscore}\n')
    file.write(f'Carla Distances:\n')
    for d in carla_distances:
        file.write(str(d))

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
