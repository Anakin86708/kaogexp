import pandas as pd
from sklearn.datasets import load_iris


class Data:
    _adult_dataset = None

    @staticmethod
    def _download_adult_dataset():
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                   'target'
                   ]
        print('Getting adult dataset, please stand by...')
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', names=columns)
        print('OK')
        return df

    @staticmethod
    def adult_dataset():
        if Data._adult_dataset is None:
            Data._adult_dataset = Data._download_adult_dataset()
        return Data._adult_dataset

    @staticmethod
    def iris_dataset():
        data = load_iris()
        df = pd.DataFrame(data.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        df['target'] = pd.Categorical.from_codes(data.target, data.target_names)
        colunas_categoricas = df.drop('target', axis=1, errors='ignore').select_dtypes(['category', 'object']).columns
        return colunas_categoricas, df
