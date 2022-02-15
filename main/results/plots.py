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


def treat_data(df: pd.DataFrame, dataset: str):
    df = df.applymap(lambda x: x / dimensions[dataset])
    df_dropped = df.T.where(df.T > 1).dropna().T
    df_ = df.T.where(df.T <= 1).dropna().T
    return df_, df_dropped


kaogexp_dists = {}

# Create dataset for each
for filename in files:
    method, dataset = filename.replace('.csv', '').split('_', 1)

    if method in ('revise', 'cem-vae'):
        continue

    if 'face' in method:
        method = method.replace('face', 'FACE')

    # Get result for KAOGExp based on dataset
    dataset_ = dataset if dataset != 'give-me-some-credit' else 'credit'
    file_result_kaogexp = os.path.join(dir_kaogexp_results, dataset_, f'metricas_{dataset_}.json')
    with open(file_result_kaogexp, 'r') as file:
        json_kaogexp = json.load(file)
        distancias_kaogexp = pd.DataFrame(list(filter(lambda x: x is not None, json_kaogexp['carla_distances'])))
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
sns.set(font_scale=1.4)
sns.set_style("whitegrid")
sns.set_context("talk")
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
                    dropped_df.to_csv(os.path.join(dir_dropped, f'{method_.lower()}_{dataset_.lower()}_dropped.csv'))

                if 'face-epsilon' in method.lower():
                    df_cols.name = 'FACE-EPS'
                elif 'wachter' in method.lower():
                    df_cols.name = 'Wachter'
                else:
                    df_cols.name = method.upper()
                df = df.append(df_cols)
            except (ValueError, TypeError):
                print(f'Error in {method} - {dataset}')

        plt.clf()
        scale_fig = 1.5
        fig = plt.figure(figsize=(6.4 * scale_fig, 4.8 * scale_fig))
        ax = fig.gca()
        plt.title(dataset.upper(), fontweight='bold')
        sns.set(font_scale=1.4)
        sns.set_style("whitegrid")
        sns.set_context("talk")
        sns.violinplot(data=df.T, orient='h', ax=ax, height=5, aspect=2)
        ax.set_xlim(0, 1)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontweight('bold')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontweight('bold')
        fig.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'{distance}_{dataset}.png'))
        plt.close()
