# Realiza o plot para comparação entre os métodos no CARLA e o KAOGExp
import json
import os.path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

working_dir = os.path.dirname(__file__)
dir_carla_results = 'E:\\Users\\silva\\PycharmProjects\\CARLA\\experiments\\results'
dir_kaogexp_results = os.path.join(working_dir, '..', 'carla_runs')
dir_dropped = os.path.join(working_dir, 'dropped')
os.makedirs(dir_dropped, exist_ok=True)
results_datasets_dirs = ['adult', 'compas', 'credit']

cols_dist = ['Distance_1', 'Distance_2', 'Distance_3', 'Distance_4']
fig_dir = os.path.join(working_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)
files = next(os.walk(dir_carla_results), (None, None, []))[2]
data = {}
# Extracted from CARLA paper
dimensions = {
    'adult': 20,
    'compas': 10,
    'give-me-some-credit': 10
}


def treat_data(df, dataset):
    df = df.applymap(lambda x: x / dimensions[dataset])
    df_dropped = df.T.where(df.T > 1).dropna().T
    df_ = df.T.where(df.T <= 1).dropna().T
    return df_, df_dropped


kaogexp_dists = {}

# Create dataset for each
for filename in files:
    method, dataset = filename.replace('.csv', '').split('_', 1)

    # Get result for KAOGExp based on dataset
    dataset_ = dataset if dataset != 'give-me-some-credit' else 'credit'
    file_result_kaogexp = os.path.join(dir_kaogexp_results, dataset_, f'metricas_{dataset_}.json')
    with open(file_result_kaogexp, 'r') as file:
        json_kaogexp = json.load(file)
        distances_ = list(filter(lambda x: x is not None, json_kaogexp['carla_distances']))
        distancias_kaogexp = pd.DataFrame(distances_)
        treat_data1, dropped_kaogexp = treat_data(distancias_kaogexp, dataset)
        treat_data1.columns = cols_dist
        kaogexp_dists[dataset] = treat_data1

        # Save dropped, if exits
        if not dropped_kaogexp.empty:
            method_ = 'kaogexp'
            dataset_ = dataset.replace('_', '-')
            dropped_kaogexp.to_csv(os.path.join(dir_dropped, f'{method_}_{dataset_}_dropped.csv'))

    df = pd.read_csv(os.path.join(dir_carla_results, filename))
    try:
        data[dataset][method] = df
    except KeyError:
        data[dataset] = {method: df}

# Iterate over distances
for distance in cols_dist:
    for dataset, methods in data.items():
        df = pd.DataFrame()
        distance_ = kaogexp_dists[dataset][distance]
        distance_.name = 'KAOGExp'
        df = df.append(distance_)
        for method, df_ in methods.items():
            try:
                df_cols, dropped_df = treat_data(pd.DataFrame([df_[distance]]), dataset)
                df_cols = df_cols.iloc[0]

                if not dropped_df.empty:
                    method_ = method.replace('_', '-')
                    dataset_ = dataset.replace('_', '-')
                    dropped_df.to_csv(os.path.join(dir_dropped, f'{method_}_{dataset_}_dropped.csv'))

                df_cols.name = method
                df = df.append(df_cols)
            except (ValueError, TypeError):
                print(f'Error in {method} - {dataset}')

        plt.clf()
        fig = plt.figure()
        ax = fig.gca()
        sns.violinplot(data=df.T, orient='h', ax=ax).set(title=dataset.upper())
        ax.set_xlim(0, 1)
        fig.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'{distance}_{dataset}.png'))
        plt.close()
