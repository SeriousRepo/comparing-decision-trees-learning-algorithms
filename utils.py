from names import DatasetNames, DatasetUrls

import pandas


def get_averages(list_of_tuples):
    sum_first = 0
    sum_second = 0
    sum_third = 0
    for i in list_of_tuples:
        sum_first += i[0]
        sum_second += i[1]
        sum_third += i[2]
    list_size = len(list_of_tuples)
    return round((sum_first / list_size), 4), round((sum_second / list_size), 4), round((sum_third / list_size), 2)


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
