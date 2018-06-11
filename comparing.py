import pandas as pd
from sklearn import model_selection
from id3 import Id3Estimator
from time import perf_counter

DatasetNames = {
    'iris': 0,
    'diabetes': 1,
    'ionosphere': 2,
    'digit': 3,
    'waveform': 4
}

SplitPartNames = {
    'X_train': 0,
    'X_test': 1,
    'y_train': 2,
    'y_test': 3
}

def getDatasetName(index):
    return list(DatasetNames.keys())[list(DatasetNames.values()).index(index)]


datasets_urls = []
datasets_urls.append('https://raw.githubusercontent.com/w4k2/data/master/datasets/iris.csv')
datasets_urls.append('https://raw.githubusercontent.com/w4k2/data/master/datasets/diabetes.csv')
datasets_urls.append('https://raw.githubusercontent.com/w4k2/data/master/datasets/ionosphere.csv')
datasets_urls.append('https://raw.githubusercontent.com/w4k2/data/master/datasets/digit.csv')
datasets_urls.append('https://raw.githubusercontent.com/w4k2/data/master/datasets/waveform.csv')

datasets = []
for url in datasets_urls:
    datasets.append(pd.read_csv(url, header=None))

Xs = []
ys = []
for single_set in datasets:
    data = single_set.values
    X = data[:, :-1]
    y = data[:, -1]
    Xs.append(X)
    ys.append(y)

k = 10
cv = model_selection.StratifiedKFold(n_splits=k)

for dataset_index in range(0, len(datasets)):
    fold = 0
    for train_indexes, test_indexes in cv.split(Xs[dataset_index], ys[dataset_index]):
        print("Zbiór {}, Fold {} ({} w TS, {} w VS)".format(getDatasetName(dataset_index), fold, len(train_indexes), len(test_indexes)))
        fold_sets = [[], [], [], []]

        fold_sets[SplitPartNames['X_train']] = Xs[dataset_index][train_indexes]
        fold_sets[SplitPartNames['y_train']] = ys[dataset_index][train_indexes]

        fold_sets[SplitPartNames['X_test']] = Xs[dataset_index][train_indexes]
        fold_sets[SplitPartNames['y_test']] = ys[dataset_index][train_indexes]

        estimator = Id3Estimator()

        start_time = perf_counter()
        estimator.fit(fold_sets[SplitPartNames['X_train']], fold_sets[SplitPartNames['y_train']])
        estimator.predict(fold_sets[SplitPartNames['X_test']])
        end_time = perf_counter()

        print(end_time-start_time)

        """
        print("predicted labels for {} dataset".format(getDatasetName(dataset_index)))
        print(estimator.predict(fold_sets[SplitPartNames['X_test']]))
        print("true labels for {} dataset".format(getDatasetName(dataset_index)))
        print(fold_sets[SplitPartNames['y_test']])
        """
        fold += 1
