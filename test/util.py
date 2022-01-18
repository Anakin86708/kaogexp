import pandas as pd
from sklearn.datasets import load_iris

from kaogexp.data.loader.DatasetFromMemory import DatasetFromMemory


class Data:
    _adult_dataset = None

    @staticmethod
    def create_new_instance_iris(tratar_na=True):
        df = Data.iris_dataset()
        colunas_categoricas = df.drop('target', axis=1, errors='ignore').select_dtypes(['category', 'object']).columns
        return DatasetFromMemory(df, colunas_categoricas, tratar_na=tratar_na)

    @staticmethod
    def create_new_instance_adult(tratar_na=True, num_sample=None):
        df = Data.adult_dataset()

        df['target'] = pd.Categorical(df['target'])
        df['target'].replace([' <=50K', ' >50K'], [0, 1], inplace=True)
        Data.define_categorical_from_object(df)
        colunas_categoricas = df.drop('target', axis=1, errors='ignore').select_dtypes(['category', 'object']).columns

        if num_sample is None:
            return DatasetFromMemory(df, colunas_categoricas, tratar_na=tratar_na)
        return DatasetFromMemory(df.sample(num_sample, random_state=42), colunas_categoricas, tratar_na=tratar_na)

    @staticmethod
    def define_categorical_from_object(df: pd.DataFrame) -> None:
        object__columns = df.select_dtypes(['object']).columns
        df[object__columns] = df[object__columns].astype('category')

    @staticmethod
    def adult_dataset():
        if Data._adult_dataset is None:
            Data._adult_dataset = Data._download_adult_dataset()
        return Data._adult_dataset.copy()

    @staticmethod
    def iris_dataset() -> pd.DataFrame:
        data = load_iris()
        df = pd.DataFrame(data.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        df['target'] = pd.Categorical(data.target)
        return df.copy()

    @staticmethod
    def _download_adult_dataset():
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                   'target'
                   ]
        print('Getting adult dataset, please stand by...')
        try:
            df = pd.read_csv('data/adult.data', names=columns)
        except:
            df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                             names=columns)
        print('OK')
        return df
