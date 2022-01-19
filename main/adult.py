# %%
# Carregar o adult dataset
import logging

import pandas as pd

from kaogexp.data.loader.DatasetFromMemory import DatasetFromMemory
from kaogexp.data.sampler.LatinSampler import LatinSampler
from kaogexp.explainer.KAOGExp import KAOGExp
from kaogexp.explainer.methods.Counterfactual import Counterfactual
from kaogexp.model.RandomForestModel import RandomForestModel

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
# dist = NewDistance(test_data.x(), test_data.nomes_colunas_categoricas)
metodo = Counterfactual
# metodo.set_metrica_distancia(dist.calculate)
# %%
explicador = KAOGExp(train_data, model, sampler)
classe_desejada = 1
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
