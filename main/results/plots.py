# Realiza o plot para comparação entre os métodos no CARLA e o KAOGExp
import os.path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

working_dir = os.path.dirname(__file__)
dir_carla_results = 'E:\\Users\\silva\\PycharmProjects\\CARLA\\experiments\\results'
dir_kaogexp_results = os.path.join(working_dir, '..', 'carla_runs')
results_datasets_dirs = ['adult', 'compas', 'credit']

cols_dist = ['Distance_1', 'Distance_2', 'Distance_3', 'Distance_4']
fig_dir = os.path.join(working_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)
files = next(os.walk(dir_carla_results), (None, None, []))[2]
data = {}
# Extracted from CARLA paper
dimensions = {
    'adult': 20,
    'compas': 8,
    'give-me-some-credit': 11
}


def treat_data(df, dataset):
    df = df.applymap(lambda x: x / dimensions[dataset])
    df_ = df.where(df <= 1).dropna()
    assert df_.max().max() <= 1 and df_.min().min() >= 0
    return df_


kaogexp_dists = {}

# Create dataset for each
for filename in files:
    method, dataset = filename.replace('.csv', '').split('_', 1)

    # Get result for KAOGExp based on dataset
    dataset_ = dataset if dataset != 'give-me-some-credit' else 'credit'
    file_result_kaogexp = os.path.join(dir_kaogexp_results, dataset_, f'{dataset_}.result')
    with open(file_result_kaogexp, 'r') as file:
        distancias_kaogexp = pd.DataFrame(eval('[' + file.readlines()[16:][0].replace('}', '}, ') + ']'))
        treat_data1 = treat_data(distancias_kaogexp, dataset)
        treat_data1.columns = cols_dist
        kaogexp_dists[dataset] = treat_data1

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
                df_cols = treat_data(pd.DataFrame([df_[distance]]), dataset).iloc[0]
                df_cols.name = method
                df = df.append(df_cols)
            except (ValueError, TypeError):
                print(f'Error in {method} - {dataset}')
            except AssertionError:
                print(f'AssertionError in {method} - {dataset}')

        plt.clf()
        fig = plt.figure()
        ax = fig.gca()
        sns.violinplot(data=df.T, orient='h', ax=ax).set(title=dataset.upper())
        ax.set_xlim(0, 1)
        fig.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'{distance}_{dataset}.png'))
        plt.close()
