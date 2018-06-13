from names import DatasetNames, DatasetUrls

import pandas


def get_datasets():
    datasets = []
    for url in DatasetUrls:
        datasets.append(pandas.read_csv(url, header=None))
    return datasets


def split_dataset(dataset):
    data = dataset.values
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def get_dataset_name(index):
    return list(DatasetNames.keys())[list(DatasetNames.values()).index(index)]
