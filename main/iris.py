# %%
# Carregar os dados padronizados de treino e teste do Iris
import pandas as pd

from kaogexp.data.loader.DatasetFromMemory import DatasetFromMemory
from kaogexp.data.sampler.LatinSampler import LatinSampler
from kaogexp.explainer.KAOGExp import KAOGExp
from kaogexp.explainer.methods.Counterfactual import Counterfactual
from kaogexp.model.RandomForestModel import RandomForestModel

categorical_columns = []
index = pd.Index(categorical_columns)
train_data = pd.concat([pd.read_csv('../data/iris/x_train.csv'), pd.read_csv('../data/iris/y_train.csv')], axis=1)
train_data = DatasetFromMemory(train_data, index)

test_data = pd.concat([pd.read_csv('../data/iris/x_test.csv'), pd.read_csv('../data/iris/y_test.csv')], axis=1)
test_data = DatasetFromMemory(test_data, index)

# %%
x = train_data.x(normalizado=True, encoded=True)
y = train_data.y()
model = RandomForestModel(x, y)

# %%
epsilon = 0.05
seed = 42
sampler = LatinSampler(epsilon=epsilon, seed=seed)

# %%
explicador = KAOGExp(train_data, model, sampler)

print('Realizando explicacao...')
explicacao = explicador.explicar(test_data.dataset().sample(10), metodo=Counterfactual)

# %%
for item in explicacao:
    try:
        print(item.instancia_modificada)
    except:
        print('Empty')
