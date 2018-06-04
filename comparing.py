import pandas as pd
from sklearn import model_selection
from id3 import Id3Estimator

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


splits = []
for index in range(0, len(datasets)):
    splits.append(model_selection.train_test_split(Xs[index], ys[index], test_size=0.333, random_state=0))


for index in range(0, len(datasets)):
    estimator = Id3Estimator()
    estimator.fit(splits[index][SplitPartNames['X_train']], splits[index][SplitPartNames['y_train']])

    print("predicted labels for {} dataset".format(getDatasetName(index)))
    print(estimator.predict(splits[index][SplitPartNames['X_test']]))
    print("true labels for {} dataset".format(getDatasetName(index)))
    print(splits[index][SplitPartNames['y_test']])

# To use
"""k = 10
cv = model_selection.StratifiedKFold(n_splits=k)

for index in range(0, len(datasets)):
    fold = 0
    for train, test in cv.split(splits[index][SplitPartName.X_train], splits[index][SplitPartName.y_train]):
        print("Zbiór {}, Fold {} ({} w TS, {} w VS)".format(index, fold, len(train), len(test)))
        print(train)
        fold += 1
"""