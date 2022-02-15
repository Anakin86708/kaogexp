import json
import logging
import operator
import os
import pickle

import pandas as pd

from main.new_distance import NewDistance
from metrics.CERScore import CERScore
from metrics.carla_metrics import CARLADistances
from metrics.dispersao import Dispersao
from metrics.proximity import Proximity
from metrics.validity import Validity


def save_tratador_and_normalizador(working_dir, name, tratador, normalizador):
    os.makedirs(os.path.join(working_dir, 'pkls'), exist_ok=True)
    with open(os.path.join(working_dir, 'pkls', f'{name}_tratador.pkl'), 'wb') as file:
        pickle.dump(tratador, file)

    with open(os.path.join(working_dir, 'pkls', f'{name}_normalizador.pkl'), 'wb') as file:
        pickle.dump(normalizador, file)


def save_counterfactuals(name, working_dir, explicacoes):
    with open(os.path.join(working_dir, 'pkls', f'{name}.pkl'), 'wb') as file:
        for item in explicacoes:
            try:
                pickle.dump(item, file)
            except AttributeError:
                print('Empty')


def compute_and_save_metrics(name, working_dir, test_dataset, test_data, explicacoes):
    logging.info("Computing metrics")
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

    # Unir todas as distancias de mesma métrica
    cerscore_carla = {}
    for distance in carla_distances[0].keys():
        op = operator.itemgetter(distance)
        cerscore_carla[distance] = cers.calcular(explicacoes,
                                                 list(map(op, filter(lambda x: x is not None, carla_distances))))

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
            'custom_distance': cerscore,
            **cerscore_carla
        },
        'carla_distances': carla_distances,
    }

    logging.info("Metrics computed")
    result_path = os.path.join(working_dir, f'metricas_{name}.json')
    logging.info(f"Saving metrics to {result_path}")
    with open(result_path, 'w') as f:
        json.dump(metricas_dict, f, indent=4)

    logging.info("Metrics saved")
