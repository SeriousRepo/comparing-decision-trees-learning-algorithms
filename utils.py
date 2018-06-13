from names import DatasetNames, DatasetUrls

import pandas
from os import system


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


def create_dataset_files(dataset, dataset_index):
    dataset_name = get_dataset_name(dataset_index)
    X, y = split_dataset(dataset)
    system('mkdir datasets/{}'.format(dataset_name))
    with open('datasets/{0}/{0}.data'.format(dataset_name), 'w') as file:
        for data_class in dataset.values:
            file.write(str(data_class[0]))
            for attribute_index in range(1, len(data_class)):
                file.write(',' + str(data_class[attribute_index]))
            file.write('\n')

    unique_labels = []

    for label in y:
        if label not in unique_labels:
            unique_labels.append(label)
    unique_labels.sort()

    with open('datasets/{0}/{0}.names'.format(dataset_name), 'w') as file:
        file.write(str(unique_labels[0]))
        for label_index in range(1, len(unique_labels)):
            file.write(', ' + str(unique_labels[label_index]))
        for attribute_index in range(1, len(X[0]) + 1):
            file.write('\n' + str(attribute_index) + ' : continuous')


def remove_directory(path):
    system('rm -r {}'.format(path))


def create_directory(path):
    system('mkdir {}'.format(path))
