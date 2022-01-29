import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

datasets = ['adult', 'compas', 'give_me_some_credit']

url_repo = 'https://raw.githubusercontent.com/carla-recourse/cf-data/main/'
current_dir = os.path.dirname(__file__)


def download_train_dataset(data_name):
    file = data_name + '_train.csv'
    url_data = url_repo + file
    url_index = url_repo + data_name + '_train_index.csv'

    _download_and_save_dataset(data_name, file, url_data, url_index)


def download_test_dataset(data_name):
    file = data_name + '_test.csv'
    url_data = url_repo + file
    url_index = url_repo + data_name + '_test_index.csv'

    _download_and_save_dataset(data_name, file, url_data, url_index)


def _download_and_save_dataset(data_name, file, url_data, url_index):
    dataset = _download_dataset(data_name, url_data, url_index)
    path = os.path.join(current_dir, file)
    dataset.to_csv(path)


def _download_dataset(data_name, url_data, url_index):
    index = _download_index(url_index)
    data = pd.read_csv(url_data)
    data.index = index
    return data


def _download_index(url_index):
    donwload = pd.read_csv(url_index, header=None)
    return pd.Index(donwload[0])


if __name__ == '__main__':
    for data_name in datasets:
        logger.info(f'Downloading {data_name}')
        download_train_dataset(data_name)
        download_test_dataset(data_name)
