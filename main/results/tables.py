import logging
import os.path
import re
from typing import Dict

import pandas as pd

# Must be changed to reflect the desired number of instances that have been explained.
num_total_instances = 100

working_dir = os.path.dirname(__file__)
dir_carla_results = 'E:\\Users\\silva\\PycharmProjects\\CARLA\\experiments\\results'
path_carla_results = 'E:\\Users\\silva\\PycharmProjects\\CARLA\\experiments\\method_metrics.csv'

distances_cols = ['Distance_1', 'Distance_2', 'Distance_3', 'Distance_4']
logger = logging.getLogger()


def main() -> Dict[str, pd.DataFrame]:
    # Load metrics results from KAOGExp
    metric_kaogexp_path = os.path.join(working_dir, '..', 'carla_runs', 'results.xlsx')
    metrics_kaogexp = pd.read_excel(metric_kaogexp_path, index_col=0)
    metrics_kaogexp.rename(index={'credit': 'give-me-some-credit'}, inplace=True)

    # Load distance results from CARLA
    cerscore = dict()
    methods = set()
    datasets = set()
    filenames_carla_results = next(os.walk(dir_carla_results), (None, None, []))[2]
    for filename in filenames_carla_results:
        method, dataset = filename.replace('.csv', '').split('_', 1)
        methods.add(method)
        datasets.add(dataset)

        # Get CERScore values for each method and dataset combination
        calculate_cerscore(cerscore, dataset, filename, method)

    # Get already computed metrics
    computed_metrics = get_computed_metrics()

    carla_data = {}
    for index, row in computed_metrics.iterrows():
        method, dataset = index.split('_')
        try:
            carla_data[dataset][method] = row
        except KeyError:
            carla_data[dataset] = {method: row}

    # Dispersão
    # CERScore
    # Validade

    # Merge data from same datasets
    results = {}
    for dataset in datasets:
        df = pd.DataFrame()

        # KAOGExp data
        kaogexp_data_ = metrics_kaogexp.loc[dataset][
            ['Proporção de validade'] + ['CERScore_Distance_' + str(i) for i in range(1, 5)]]
        kaogexp_data_.name = 'KAOGExp'
        df = df.append(kaogexp_data_)
        df.rename(columns={'Proporção de validade': 'Success_Rate'}, inplace=True)

        # Add CARLA data and CERScore
        for method in methods:
            try:
                cerscore_dataset_method_: pd.Series = cerscore[dataset][method]
                cerscore_dataset_method_.rename(
                    index={col: 'CERScore_' + col for col in cerscore_dataset_method_.index}, inplace=True
                )
                cerscore_dataset_method_['Success_Rate'] = carla_data[dataset][method].loc['Success_Rate']
                cerscore_dataset_method_.name = method
                df = df.append(cerscore_dataset_method_)
            except KeyError:
                logger.error('Dataset {} or method {} not found'.format(dataset, method))
                pass

        results[dataset] = df

    return results


def calculate_cerscore(cerscore, dataset, filename, method):
    df = pd.read_csv(os.path.join(dir_carla_results, filename))
    cerscore_ = calculate_cerscore_for_data(df)
    try:
        if cerscore_ is not None:
            cerscore[dataset][method] = cerscore_
    except KeyError:
        if cerscore_ is not None:
            cerscore[dataset] = {method: cerscore_}


def calculate_cerscore_for_data(df: pd.DataFrame):
    try:
        # TODO: inserir prob e inverso da distância
        prob = df['Probability']
        dist = (1 / df[distances_cols])
        return dist.apply(lambda x: x * prob).sum()
    except (TypeError, KeyError):
        logger.error('Empty dataframe for CERScore')
        return None


def get_computed_metrics():
    computed_metrics = pd.read_csv(path_carla_results, index_col=0)
    # Clean indexes
    indexes = []
    for index in computed_metrics.index:
        indexes.append(re.sub('[()\'\\s]', '', index).replace('_', '-').replace(',', '_'))
    computed_metrics.index = indexes
    return computed_metrics


if __name__ == '__main__':
    results = main()
    tables_dir = os.path.join(working_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    for name, df in results.items():
        df.to_excel(os.path.join(tables_dir, f'{name}.xlsx'))
