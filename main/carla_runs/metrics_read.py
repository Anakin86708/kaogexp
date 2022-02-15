# Create graph for dispersion and proximity
import json
import logging
import os.path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import Series, DataFrame

logger = logging.getLogger()

work_dir = os.path.dirname(__file__)
names = ['adult', 'compas', 'credit']

columns = ['Proporção de validade', 'CERScore_custom_distance', 'CERScore_Distance_1', 'CERScore_Distance_2',
           'CERScore_Distance_3', 'CERScore_Distance_4']


def generate_stats(name) -> tuple[Series, Series, Series, DataFrame]:
    path = os.path.join(work_dir, name, f'metricas_{name}.json')
    carla_distances, cerscore, prop_validade, proximidade, dispersao = get_values_from_file(path)

    df_carla_distances = pd.DataFrame(carla_distances)

    df_carla_distances = df_carla_distances.applymap(lambda x: x / 20)

    df_stats_carla = pd.DataFrame(carla_distances).describe()
    df_stats_carla.to_excel(os.path.join(work_dir, name, name + '_stats_carla.xlsx'))

    results = pd.Series({f'CERScore_{k}': v for k, v in cerscore.items()}, name=name, index=columns, dtype=float)
    results['Proporção de validade'] = prop_validade

    return results, proximidade, dispersao, df_carla_distances


def get_values_from_file(path):
    with open(path, 'r') as file:
        metricas = json.load(file)

    carla_distances = metricas['carla_distances']
    cerscore = metricas['cerscore']
    prop_validade = metricas['validade']['proporcao_validade']
    proximidade = pd.Series(metricas['proximidade']['proximidades'], name=name)
    dispersao = pd.Series(metricas['dispersao'], name=name)

    return list(filter(lambda x: x is not None, carla_distances)), cerscore, prop_validade, proximidade, dispersao


if __name__ == '__main__':
    df_results = pd.DataFrame(columns=columns)
    proximidades = pd.DataFrame()
    dispersoes = pd.DataFrame()
    distancias_carla = {}
    for name in names:
        try:
            stats, proximidade, dispersao, dist_carla = generate_stats(name)
            df_results = df_results.append(stats)
            proximidades[proximidade.name] = proximidade
            dispersoes[dispersao.name] = dispersao
            distancias_carla[name] = dist_carla
        except FileNotFoundError:
            logger.error(f'Unable to find results file for {name}')
            continue

    df_results.to_excel(os.path.join(work_dir, 'results.xlsx'))

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
