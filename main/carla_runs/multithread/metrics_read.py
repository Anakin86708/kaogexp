import os.path
from typing import Any

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import Series

work_dir = os.path.dirname(__file__)
names = ['adult', 'compas', 'credit']

columns = ['Proporção de validade', 'CERScore', 'Média proximidade', 'Desvio padrão proximidade',
           'Mínima proximidade', 'Máxima proximidade']


def generate_stats(name) -> tuple[Series, Series, Series, Any]:
    path = os.path.join(work_dir, name, name + '.result')
    carla_distances, cerscore, prop_validade, proximidade, dispersao = get_values_from_file(path)

    df_stats_carla = pd.DataFrame(carla_distances).describe()
    df_stats_carla.to_excel(os.path.join(work_dir, name, name + '_stats_carla.xls'))

    stats_proximidade = proximidade.describe()
    results = pd.Series(name=name, index=columns, dtype=float)
    results['Proporção de validade'] = prop_validade
    results['CERScore'] = cerscore
    results['Média proximidade'] = stats_proximidade['mean']
    results['Desvio padrão proximidade'] = stats_proximidade['std']
    results['Mínima proximidade'] = stats_proximidade['min']
    results['Máxima proximidade'] = stats_proximidade['max']

    return results, proximidade, dispersao, carla_distances


def get_values_from_file(path):
    with open(path, 'r', encoding='UTF-8') as file:
        line = file.readline()
        # validade
        validade = eval(line.replace('Validade: ', ''))

        # proporção validade
        prop_validade = eval(file.readline().replace('Proporção de validade: ', ''))

        # dispresão
        dispersao = pd.Series(eval(file.readline().replace('Dispersão: ', '')), name=name)

        # proximidade
        proximidade = pd.Series(eval(file.readline().replace('Proximidade: ', '')), name=name)

        for _ in range(10):
            file.readline()

        # CERScore
        cerscore = eval(file.readline().replace('CERScore: ', ''))

        # carla distances
        file.readline()
        carla_distances = eval('[' + file.readline().replace('}', '}, ') + ']')
    return carla_distances, cerscore, prop_validade, proximidade, dispersao


if __name__ == '__main__':
    df_results = pd.DataFrame(columns=columns)
    proximidades = pd.DataFrame()
    dispersoes = pd.DataFrame()
    distancias_carla = {}
    for name in names:
        stats, proximidade, dispersao, dist_carla = generate_stats(name)
        df_results = df_results.append(stats)
        proximidades[proximidade.name] = proximidade
        dispersoes[dispersao.name] = dispersao
        distancias_carla[name] = dist_carla

    df_results.to_excel(os.path.join(work_dir, 'results.xls'))

    # plots
    plt.clf()
    ax = sns.violinplot(data=proximidades, orient='h', cut=0)
    plt.savefig('proximity.png')

    plt.clf()
    min_ = dispersoes.min().min() - 1
    max_ = dispersoes.max().max()
    ax = sns.displot(data=dispersoes, multiple="dodge", binrange=(min_, max_))
    plt.savefig('dispersion.png')
    plt.clf()

    path_join = os.path.join(work_dir, 'carla')
    os.makedirs(path_join, exist_ok=True)
    for name, data in distancias_carla.items():
        ax = sns.violinplot(data=pd.DataFrame(data), orient='h', cut=0).set(title=name)
        plt.savefig(os.path.join(path_join, f'{name}_carla_dist.png'))
        plt.clf()
