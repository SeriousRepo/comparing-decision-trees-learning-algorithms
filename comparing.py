import pandas as pd

datasets_urls = []
datasets_urls.append(('iris', 'https://raw.githubusercontent.com/w4k2/data/master/datasets/iris.csv'))
datasets_urls.append(('diabetes', 'https://raw.githubusercontent.com/w4k2/data/master/datasets/diabetes.csv'))
datasets_urls.append(('ionosphere', 'https://raw.githubusercontent.com/w4k2/data/master/datasets/ionosphere.csv'))
datasets_urls.append(('digit', 'https://raw.githubusercontent.com/w4k2/data/master/datasets/digit.csv'))
datasets_urls.append(('waveform', 'https://raw.githubusercontent.com/w4k2/data/master/datasets/waveform.csv'))

datasets = []
for url in datasets_urls:
    datasets.append((url[0], pd.read_csv(url[1], header=None)))

Xs = []
ys = []
for single_set in datasets:
    data = single_set[1].values
    X = data[:, :-1]
    y = data[:, -1]
    Xs.append(X)
    ys.append(y)

