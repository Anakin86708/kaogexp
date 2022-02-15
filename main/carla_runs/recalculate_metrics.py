# Used to recalculate metrics for a given run, using .pkl objects
import logging
import os.path
import pickle

import pandas as pd

from data.loader import ColunaYSingleton
from data.loader.DatasetFromMemory import DatasetFromMemory
from main.carla_runs.util import compute_and_save_metrics

working_dir = os.path.dirname(__file__)

logger = logging.getLogger(__name__)


def main():
    datasets_names = ['adult', 'compas', 'credit']
    data_paths = {
        'adult': os.path.join(working_dir, '../../data/carla_data/adult_test.csv'),
        'compas': os.path.join(working_dir, '../../data/carla_data/compas_test.csv'),
        'credit': os.path.join(working_dir, '../../data/carla_data/give_me_some_credit_test.csv')
    }
    cat_cols = {
        'adult': pd.Index(['workclass', 'marital-status', 'occupation', 'relationship',
                           'race', 'sex', 'native-country']),
        'compas': pd.Index(['c_charge_degree', 'race', 'sex']),
        'credit': pd.Index([])
    }
    col_y = {
        'adult': 'income',
        'compas': 'score',
        'credit': 'SeriousDlqin2yrs'
    }

    for dataset_name in datasets_names:
        logger.info(f'Calculating metrics for {dataset_name}')
        try:
            explicacoes = []
            # load pickle
            counterfactual_pkl_path = os.path.join(working_dir, dataset_name, 'pkls', f'{dataset_name}.pkl')
            with open(counterfactual_pkl_path, 'rb') as file:
                logger.info(f'Loading {counterfactual_pkl_path}')
                while True:
                    try:
                        explicacoes.append(pickle.load(file))
                    except EOFError:
                        break
                logger.info(f'Loaded {len(explicacoes)} counterfactuals for {dataset_name}')

            # load test dataset
            test_dataset = pd.read_csv(data_paths[dataset_name], index_col=0)
            test_dataset.columns = test_dataset.columns.str.strip()

            # create data
            ColunaYSingleton().NOME_COLUNA_Y = col_y[dataset_name]
            test_data = DatasetFromMemory(test_dataset, cat_cols[dataset_name])

            compute_and_save_metrics(dataset_name, os.path
                                     .join(working_dir, dataset_name), test_dataset, test_data, explicacoes)
        finally:
            del explicacoes

    logger.info('End')


if __name__ == '__main__':
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    main()
